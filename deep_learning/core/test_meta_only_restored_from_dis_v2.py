import os
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.stateless import functional_call
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from deep_learning.core.config import Config
from deep_learning.core.meta_db_loader import load_cycle_db_for_meta
from deep_learning.core.data_seq_group import group_data_by_battery_from_df
from deep_learning.core.data import MetaBatteryDataset
from deep_learning.core.models import MultiTaskRULModel
from deep_learning.core.meta_utils import (
    DEVICE,
    ParamVectorizer,
    bmaml_inner,
    compute_neg_logp,
    make_leaf_thetas,
)
from deep_learning.core.scalers import unscale_rul_array
from deep_learning.core.train_meta import meta_collate_fn


def meta_evaluate_pos_only(model, vecizer, meta_thetas, loader, c: Config, max_rul_train: float):
    """
    Few-shot evaluation on meta-test tasks.
    - RUL>0 인 구간만 사용해서 RMSE / MAE / R2 계산
    - R2는 [-1, 1] 로 클리핑해서 극단적인 음수(-10 등) 방지
    """
    model.eval()

    all_y = []
    all_pred = []
    all_mse = []
    all_calib = []
    all_aux = []

    for tasks in loader:
        for task in tasks:
            with torch.enable_grad():
                theta0 = make_leaf_thetas(meta_thetas, detach=True)
                theta_p, _, eff, _, _ = bmaml_inner(
                    model=model,
                    vecizer=vecizer,
                    theta0=theta0,
                    task=task,
                    c=c,
                    detach_theta0=True,
                    return_losses=False,
                )

            if (not eff) or (theta_p is None):
                continue

            with torch.no_grad():
                q_seq = torch.nan_to_num(task["q_seq"].to(DEVICE), nan=0.0, posinf=0.0, neginf=0.0)
                q_sum = torch.nan_to_num(task["q_sum"].to(DEVICE), nan=0.0, posinf=0.0, neginf=0.0)
                q_rul = torch.nan_to_num(task["q_rul"].to(DEVICE), nan=0.0, posinf=0.0, neginf=0.0)

                preds = []
                for tp in theta_p:
                    params = vecizer.vector_to_params(tp)
                    out, _, _, _ = functional_call(model, params, (q_seq, q_sum))
                    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
                    preds.append(out.squeeze(-1))

                if len(preds) == 0:
                    continue

                mean_pred = torch.stack(preds).mean(dim=0)

                y_np_scaled = q_rul.detach().cpu().numpy()
                pred_np_scaled = mean_pred.detach().cpu().numpy()

                y = unscale_rul_array(y_np_scaled, c.rul_mode, max_rul_train)
                pred = unscale_rul_array(pred_np_scaled, c.rul_mode, max_rul_train)

                all_y.append(y)
                all_pred.append(pred)

                s_seq = torch.nan_to_num(task["s_seq"].to(DEVICE), nan=0.0, posinf=0.0, neginf=0.0)
                s_sum = torch.nan_to_num(task["s_sum"].to(DEVICE), nan=0.0, posinf=0.0, neginf=0.0)
                s_rul = torch.nan_to_num(task["s_rul"].to(DEVICE), nan=0.0, posinf=0.0, neginf=0.0)

                cycles = task.get("cycles")
                if cycles is not None:
                    cycles = cycles.to(DEVICE)

                seq_union = torch.cat([s_seq, q_seq], dim=0)
                sum_union = torch.cat([s_sum, q_sum], dim=0)
                rul_union = torch.cat([s_rul, q_rul], dim=0)

                neg_logp, _, _, _, mse, calib, aux = compute_neg_logp(
                    model=model,
                    vecizer=vecizer,
                    theta_particles=theta_p,
                    seq_x=seq_union,
                    sum_x=sum_union,
                    y=rul_union,
                    cycles=cycles,
                    c=c,
                )
                all_mse.append(float(mse.item()))
                all_calib.append(float(calib.item()))
                all_aux.append(float(aux.item()))

    if len(all_y) == 0:
        return (1000000.0, -1000000.0, 1000000.0, 0.0, 0.0, 0.0)

    y = np.concatenate(all_y)
    pred = np.concatenate(all_pred)

    mask = y > 0
    if mask.sum() >= 2:
        y_eval = y[mask]
        pred_eval = pred[mask]
    else:
        y_eval = y
        pred_eval = pred

    rmse = float(np.sqrt(mean_squared_error(y_eval, pred_eval)))
    mae = float(mean_absolute_error(y_eval, pred_eval))

    if (len(y_eval) > 1) and (np.var(y_eval) > 1e-8):
        r2 = float(r2_score(y_eval, pred_eval))
        r2 = float(np.clip(r2, -1.0, 1.0))
    else:
        r2 = 0.0

    mse_avg = float(np.mean(all_mse)) if all_mse else 0.0
    calib_avg = float(np.mean(all_calib)) if all_calib else 0.0
    aux_avg = float(np.mean(all_aux)) if all_aux else 0.0

    return (rmse, r2, mae, mse_avg, calib_avg, aux_avg)


def main():
    ckpt_path = "./core_checkpoints_nasa/nasa_bmaml_best_re_3.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=DEVICE)

    cfg = Config()
    if "config" in ckpt:
        cfg.__dict__.update(ckpt["config"])

    cfg.checkpoint_dir = os.path.dirname(ckpt_path)

    df, target_col, _, bid_col, cyc_col = load_cycle_db_for_meta(cfg)

    feature_cols = ckpt.get("feature_cols")
    if feature_cols is None:
        raise RuntimeError("feature_cols not found in checkpoint.")

    grouped = group_data_by_battery_from_df(
        df=df,
        cfg=cfg,
        target_col=target_col,
        feature_cols=feature_cols,
        bid_col=bid_col,
        cyc_col=cyc_col,
    )

    train_bids = ["B0054", "B0046", "B0005", "B0044", "B0034", "B0007", "B0006"]
    val_bids = ["B0055", "B0033"]
    test_bids = ["B0043", "B0048", "B0018", "B0042"]

    def check_exist(name: str, bids: List[str]):
        missing = [b for b in bids if b not in grouped]
        if missing:
            print(f"[WARN] {name} batteries not found in grouped data: {missing}")
            return
        return

    check_exist("TRAIN", train_bids)
    check_exist("VAL", val_bids)
    check_exist("TEST", test_bids)

    print(f"[META-TEST ONLY] train_batteries = {train_bids}")
    print(f"[META-TEST ONLY] val_batteries   = {val_bids}")
    print(f"[META-TEST ONLY] test_batteries  = {test_bids}")

    seq_scaler = ckpt["seq_scaler"]
    sum_scaler = ckpt["sum_scaler"]
    max_rul_train = ckpt["max_rul_train"]

    test_ds = MetaBatteryDataset(
        grouped_data=grouped,
        battery_ids=test_bids,
        c=cfg,
        seq_scaler=seq_scaler,
        sum_scaler=sum_scaler,
        mode="val",
        max_rul_train=max_rul_train,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.meta_batch_size,
        shuffle=False,
        collate_fn=meta_collate_fn,
    )

    if len(test_ds) == 0:
        raise RuntimeError("No samples in test dataset.")

    any_bid = test_bids[0]
    if any_bid not in grouped:
        any_bid = sorted(grouped.keys())[0]

    sd = grouped[any_bid]["seq"].shape[-1]
    sm = grouped[any_bid]["sum"].shape[-1]

    model = MultiTaskRULModel(sd, sm, cfg).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])

    meta_thetas = [t.to(DEVICE) for t in ckpt["meta_thetas"]]

    vecizer = ParamVectorizer(model)

    rmse_t, r2_t, mae_t, mse_t, calib_t, aux_t = meta_evaluate_pos_only(
        model=model,
        vecizer=vecizer,
        meta_thetas=meta_thetas,
        loader=test_loader,
        c=cfg,
        max_rul_train=max_rul_train,
    )

    print(
        f"[META-TEST ONLY] RMSE={rmse_t:.3f} | MAE={mae_t:.3f} | R2={r2_t:.3f} | MSE={mse_t:.4f} | CAL={calib_t:.4f} | AUX={aux_t:.4f}"
    )
    return


if __name__ == "__main__":
    main()
