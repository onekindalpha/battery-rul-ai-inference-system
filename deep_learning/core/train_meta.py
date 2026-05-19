# deep_learning/core/train_meta.py
# BMAML-SVGD + CEEMDAN-Transformer-DNN 백본 메타 트레이닝 엔트리포인트
# + (옵션) Weights & Biases 로깅 지원

import os
import random
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.stateless import functional_call
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    import wandb
except ImportError:
    wandb = None

from deep_learning.core.config import Config
from deep_learning.core.meta_db_loader import load_cycle_db_for_meta
from deep_learning.core.data_seq_group import group_data_by_battery_from_df
from deep_learning.core.data import MetaBatteryDataset
from deep_learning.core.scalers import CustomRobustScaler3D, unscale_rul_array
from deep_learning.core.models import MultiTaskRULModel
from deep_learning.core.meta_utils import (
    DEVICE,
    ParamVectorizer,
    bmaml_inner,
    compute_neg_logp,
    make_leaf_thetas,
)

# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def meta_collate_fn(batch):
    """MetaBatteryDataset 가 이미 task(dict)를 반환하므로 그대로 리스트로 전달."""
    return batch


def fit_scalers(grouped, train_bids, c: Config):
    """시계열 / summary scaler 를 train 배터리 기준으로 fit."""
    seq_list, sum_list = [], []

    for bid in train_bids:
        data = grouped[bid]
        n = len(data["seq"])
        fit_len = int(n * c.scaler_fit_ratio)
        fit_len = max(fit_len, c.k_shot + c.q_query + 1)
        fit_len = min(fit_len, n)

        seq_list.append(data["seq"][:fit_len])
        sum_list.append(data["sum"][:fit_len])

    seq_arr = np.concatenate(seq_list, axis=0)
    sum_arr = np.concatenate(sum_list, axis=0)

    seq_scaler = CustomRobustScaler3D(p=90, min_iqr=1e-3, clip_value=c.seq_clip)
    seq_scaler.fit(seq_arr)

    sum_scaler = RobustScaler()
    sum_scaler.fit(sum_arr)

    # RUL scaling 에 사용할 train set 기준 max_rul
    max_rul_train = max(float(grouped[b]["max_rul"]) for b in train_bids)
    return seq_scaler, sum_scaler, max_rul_train


def init_model_and_particles(sd: int, sm: int, c: Config):
    """모델 + 메타 파티클(theta) 초기화.
    CEEMDAN-Transformer-DNN supervised backbone 체크포인트가 있으면,
    그 가중치를 먼저 로드한 뒤 작은 노이즈를 주어 meta 파티클을 생성한다.
    """
    model = MultiTaskRULModel(sd, sm, c).to(DEVICE)

    # backbone checkpoint 경로 결정
    ckpt_dir = getattr(c, "checkpoint_dir", "./core_checkpoints")
    default_backbone = os.path.join(
        ckpt_dir,
        f"{c.dataset_source}_backbone_ceemdan_trf_dnn_re_(2).pt",
    )
    backbone_ckpt_path = getattr(c, "backbone_ckpt_path", default_backbone)

    if backbone_ckpt_path and os.path.exists(backbone_ckpt_path):
        try:
            ckpt = torch.load(backbone_ckpt_path, map_location=DEVICE)
            if isinstance(ckpt, dict):
                state = ckpt.get("model_state") or ckpt.get("model_state_dict") or ckpt
            else:
                state = ckpt
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(f"[INIT] Loaded backbone from {backbone_ckpt_path}")
            if missing:
                print(f"[INIT]  missing keys: {missing}")
            if unexpected:
                print(f"[INIT]  unexpected keys: {unexpected}")
        except Exception as e:
            print(f"[INIT WARN] Failed to load backbone ckpt ({backbone_ckpt_path}): {e}")
    else:
        print(f"[INIT] No backbone checkpoint found at {backbone_ckpt_path}, using random init.")

    vecizer = ParamVectorizer(model)

    # base 파라미터에서 작은 노이즈를 준 particle theta 생성
    base_vec = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
    std = float(getattr(c, "particle_init_std", 0.1))

    meta_thetas: List[torch.Tensor] = []
    for _ in range(c.num_particles):
        params_vec = base_vec + std * torch.randn_like(base_vec)
        params_vec.requires_grad_(True)
        meta_thetas.append(params_vec)

    return model, vecizer, meta_thetas


# ---------------------------------------------------------------------
# Meta training / evaluation
# ---------------------------------------------------------------------

def meta_train_epoch(model, vecizer, meta_thetas, outer_opt, loader, c: Config):
    model.train()
    outer_opt.zero_grad(set_to_none=True)

    meta_loss = torch.tensor(0.0, device=DEVICE)
    effective_tasks = 0

    for tasks in loader:  # tasks: List[dict]
        for task in tasks:
            theta_p, theta_s, eff, neg_q_n, neg_union = bmaml_inner(
                model,
                vecizer,
                meta_thetas,
                task,
                c,
                detach_theta0=False,
                return_losses=True,
            )
            if (not eff) or (neg_q_n is None):
                continue

            meta_loss = meta_loss + neg_q_n
            effective_tasks += 1

    if effective_tasks == 0:
        outer_opt.zero_grad(set_to_none=True)
        return 0.0, 0

    meta_loss = meta_loss / effective_tasks
    meta_loss.backward()

    clip_val = float(getattr(c, "outer_clip", 1.0))
    clip_grad_norm_(meta_thetas, clip_val)

    outer_opt.step()
    outer_opt.zero_grad(set_to_none=True)

    return float(meta_loss.item()), effective_tasks


def meta_evaluate(model, vecizer, meta_thetas, loader, c: Config, max_rul_train: float):
    """Few-shot evaluation on meta-val / meta-test tasks."""
    model.eval()

    all_y, all_pred = [], []
    all_mse, all_calib, all_aux = [], [], []

    for tasks in loader:
        for task in tasks:
            # 1) meta-thetas 에서 inner adaptation (detach_theta0=True: gradient는 사용 안함)
            with torch.enable_grad():
                theta0 = make_leaf_thetas(meta_thetas, detach=True)
                theta_p, _, eff, _, _ = bmaml_inner(
                    model,
                    vecizer,
                    theta0,
                    task,
                    c,
                    detach_theta0=True,
                    return_losses=False,
                )

            if (not eff) or (theta_p is None):
                continue

            # 2) adapted theta_p 에 대해 query set 예측
            with torch.no_grad():
                q_seq = torch.nan_to_num(
                    task["q_seq"].to(DEVICE),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                q_sum = torch.nan_to_num(
                    task["q_sum"].to(DEVICE),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                q_rul = torch.nan_to_num(
                    task["q_rul"].to(DEVICE),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )

                preds = []
                for tp in theta_p:
                    params = vecizer.vector_to_params(tp)
                    out, _, _, _ = functional_call(model, params, (q_seq, q_sum))
                    out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
                    preds.append(out.squeeze(-1))

                if len(preds) == 0:
                    continue

                mean_pred = torch.stack(preds).mean(dim=0)

                # scale된 RUL → 실제 RUL 단위로 복원
                y_np_scaled = q_rul.detach().cpu().numpy()
                pred_np_scaled = mean_pred.detach().cpu().numpy()
                y = unscale_rul_array(y_np_scaled, c.rul_mode, max_rul_train)
                pred = unscale_rul_array(pred_np_scaled, c.rul_mode, max_rul_train)

                all_y.append(y)
                all_pred.append(pred)

                # 3) 세부 loss (mse / calib / aux) 모니터링용
                s_seq = torch.nan_to_num(
                    task["s_seq"].to(DEVICE),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                s_sum = torch.nan_to_num(
                    task["s_sum"].to(DEVICE),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                s_rul = torch.nan_to_num(
                    task["s_rul"].to(DEVICE),
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
                cycles = task.get("cycles")
                if cycles is not None:
                    cycles = cycles.to(DEVICE)

                seq_union = torch.cat([s_seq, q_seq], dim=0)
                sum_union = torch.cat([s_sum, q_sum], dim=0)
                rul_union = torch.cat([s_rul, q_rul], dim=0)

                neg_logp, _, _, _, mse, calib, aux = compute_neg_logp(
                    model,
                    vecizer,
                    theta_p,
                    seq_union,
                    sum_union,
                    rul_union,
                    cycles,
                    c,
                )
                all_mse.append(float(mse.item()))
                all_calib.append(float(calib.item()))
                all_aux.append(float(aux.item()))

    if len(all_y) == 0:
        return 1e6, -1e6, 1e6, 0.0, 0.0, 0.0

    if len(all_y) == 0:
        return 1e6, -1e6, 1e6, 0.0, 0.0, 0.0

    if len(all_y) == 0:
        return 1e6, -1e6, 1e6, 0.0, 0.0, 0.0

    y = np.concatenate(all_y)
    pred = np.concatenate(all_pred)

    # -----------------------------
    # 🔧 수정 1: RUL>0 구간만 평가
    # -----------------------------
    mask = y > 0
    if mask.sum() >= 2:
        y_eval = y[mask]
        pred_eval = pred[mask]
    else:
        # 거의 다 0이면 그냥 전체로 fallback
        y_eval = y
        pred_eval = pred

    rmse = float(np.sqrt(mean_squared_error(y_eval, pred_eval)))
    mae  = float(mean_absolute_error(y_eval, pred_eval))

    # -----------------------------
    # 🔧 수정 2: R² 안정화 (clip 범위도 축소)
    # -----------------------------
    if len(y_eval) > 1 and np.var(y_eval) > 1e-8:
        r2 = float(r2_score(y_eval, pred_eval))
        # 너무 극단적인 음수는 의미가 없으니 [-1, 1] 정도로만 클립
        r2 = float(np.clip(r2, -1.0, 1.0))
    else:
        r2 = 0.0

    mse_avg   = float(np.mean(all_mse))   if all_mse   else 0.0
    calib_avg = float(np.mean(all_calib)) if all_calib else 0.0
    aux_avg   = float(np.mean(all_aux))   if all_aux   else 0.0

    return rmse, r2, mae, mse_avg, calib_avg, aux_avg


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    cfg = Config()
    set_seed(cfg.RANDOM_SEED)
    # ======================================================
    # (선택) 메타 체크포인트에서 warm-start 재개
    #  - 예: epoch 006 에서 저장된 nasa_bmaml_best_re.pt
    #  - "" 로 두면 resume 안 함
    # ======================================================
    resume_ckpt_path = ""  # 여기 경로 바꿔 써
    resume_state = None
    if resume_ckpt_path and os.path.exists(resume_ckpt_path):
        print(f"[RESUME] Loading meta checkpoint from {resume_ckpt_path}")
        resume_state = torch.load(resume_ckpt_path, map_location=DEVICE)

        # ckpt 안에 config 있으면 덮어쓰기 (dataset_source 등 맞추기 위함)
        if "config" in resume_state:
            cfg.__dict__.update(resume_state["config"])

        # 체크포인트 디렉토리도 맞춰주기
        cfg.checkpoint_dir = os.path.dirname(resume_ckpt_path)
    else:
        print("[RESUME] No resume checkpoint found (or path empty). Start from scratch.")
    # WandB 설정 (옵션)
    use_wandb = bool(getattr(cfg, "use_wandb", False))
    wandb_run = None
    if use_wandb:
        if wandb is None:
            raise ImportError("cfg.use_wandb=True 인데 wandb 패키지가 설치되어 있지 않습니다. `pip install wandb` 필요.")
        project = getattr(cfg, "wandb_project", "battery_rul_bmaml_svgd")
        group = getattr(cfg, "wandb_group", f"bmaml_svgd_{cfg.dataset_source}")
        mode = getattr(cfg, "wandb_mode", "online")  # "online" / "offline" / "disabled"
        run_name = getattr(cfg, "wandb_run_name", f"{cfg.dataset_source}_meta")

        wandb_run = wandb.init(
            project=project,
            group=group,
            name=run_name,
            mode=mode,
            config=cfg.__dict__,
        )

    # 1) 공통 baseline 으로부터 cycle-level DB 로드
    df, target_col, feature_cols, bid_col, cyc_col = load_cycle_db_for_meta(cfg)
    # (resume 시) ckpt 안에 feature_cols가 있으면 그걸 우선 사용
    if resume_state is not None and "feature_cols" in resume_state:
        feature_cols = resume_state["feature_cols"]
    # 2) 배터리 단위 그룹핑
    grouped = group_data_by_battery_from_df(
        df=df,
        cfg=cfg,
        target_col=target_col,
        feature_cols=feature_cols,
        bid_col=bid_col,
        cyc_col=cyc_col,
    )

    all_bids = sorted(grouped.keys())
    if len(all_bids) < 3:
        raise RuntimeError(f"Too few batteries ({len(all_bids)}) for meta-training.")

    rng = np.random.RandomState(cfg.RANDOM_SEED)
    rng.shuffle(all_bids)

    n_total = len(all_bids)
    n_train = max(1, int(n_total * 0.6))
    n_val = max(1, int(n_total * 0.2))
    n_test = max(1, n_total - n_train - n_val)

    # ======================================
    # (1) resume 이면 ckpt의 split 재사용
    # ======================================
    if resume_state is not None and "train_bids" in resume_state:
        train_bids = resume_state["train_bids"]
        val_bids   = resume_state.get("val_bids", [])
        test_bids  = resume_state.get("test_bids", [])
        print(f"[RESUME] Using splits from checkpoint.")
    else:
        # ======================================
        # (2) 아니면 기존 로직대로 새로 split
        # ======================================
        train_bids = all_bids[:n_train]
        val_bids   = all_bids[n_train:n_train + n_val]
        test_bids  = all_bids[n_train + n_val: n_train + n_val + n_test]

    print(f"[SPLIT] train={len(train_bids)}, val={len(val_bids)}, test={len(test_bids)}")
    # 3) Scaler fit
    if resume_state is not None:
        seq_scaler    = resume_state["seq_scaler"]
        sum_scaler    = resume_state["sum_scaler"]
        max_rul_train = resume_state["max_rul_train"]
        print("[RESUME] Loaded scalers and max_rul_train from checkpoint.")
    else:
        seq_scaler, sum_scaler, max_rul_train = fit_scalers(grouped, train_bids, cfg)
    # 4) MetaBatteryDataset / DataLoader
    train_ds = MetaBatteryDataset(
        grouped_data=grouped,
        battery_ids=train_bids,
        c=cfg,
        seq_scaler=seq_scaler,
        sum_scaler=sum_scaler,
        mode="train",
        max_rul_train=max_rul_train,
    )
    val_ds = MetaBatteryDataset(
        grouped_data=grouped,
        battery_ids=val_bids,
        c=cfg,
        seq_scaler=seq_scaler,
        sum_scaler=sum_scaler,
        mode="val",
        max_rul_train=max_rul_train,
    )
    test_ds = MetaBatteryDataset(
        grouped_data=grouped,
        battery_ids=test_bids,
        c=cfg,
        seq_scaler=seq_scaler,
        sum_scaler=sum_scaler,
        mode="val",
        max_rul_train=max_rul_train,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.meta_batch_size,
        shuffle=True,
        collate_fn=meta_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.meta_batch_size,
        shuffle=False,
        collate_fn=meta_collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.meta_batch_size,
        shuffle=False,
        collate_fn=meta_collate_fn,
    )

    # 5) 모델 / 파티클 / 옵티마이저 초기화
    sd = grouped[train_bids[0]]["seq"].shape[-1]
    sm = grouped[train_bids[0]]["sum"].shape[-1]

    if resume_state is not None:
        # 모델 생성 후 ckpt 가중치 로드
        model = MultiTaskRULModel(sd, sm, cfg).to(DEVICE)
        model.load_state_dict(resume_state["model_state"])
        vecizer = ParamVectorizer(model)

        # meta_thetas 복원 (leaf tensor + grad 켜기)
        meta_thetas = []
        for t in resume_state["meta_thetas"]:
            v = t.to(DEVICE).detach().clone()
            v.requires_grad_(True)
            meta_thetas.append(v)

        print("[RESUME] Model + meta_thetas loaded from checkpoint.")
    else:
        # 처음부터 학습
        model, vecizer, meta_thetas = init_model_and_particles(sd, sm, cfg)
    outer_opt = torch.optim.AdamW(
        meta_thetas,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # 6) 학습 루프
    best_rmse = 1e9
    best_state = None
    patience = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss, eff = meta_train_epoch(
            model, vecizer, meta_thetas, outer_opt, train_loader, cfg
        )
        rmse, r2, mae, mse_avg, calib_avg, aux_avg = meta_evaluate(
            model, vecizer, meta_thetas, val_loader, cfg, max_rul_train
        )

        print(
            f"[Epoch {epoch:03d}] "
            f"TrainLoss={train_loss:.4f} | EffTasks={eff} | "
            f"RMSE={rmse:.3f} | MAE={mae:.3f} | R2={r2:.3f} | "
            f"MSE={mse_avg:.4f} | CAL={calib_avg:.4f} | AUX={aux_avg:.4f}"
        )

        if use_wandb and wandb_run is not None:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "eff_tasks": eff,
                    "val_rmse": rmse,
                    "val_mae": mae,
                    "val_r2": r2,
                    "val_mse": mse_avg,
                    "val_calib": calib_avg,
                    "val_aux": aux_avg,
                }
            )

        if rmse < best_rmse:
            best_rmse = rmse
            patience = 0

            # 메모리 상의 best_state 갱신
            best_state = {
                "model": model.state_dict(),
                "meta_thetas": [t.detach().cpu() for t in meta_thetas],
                "cfg": cfg,
            }

            # 🔥 여기서 바로 디스크에도 저장
            ckpt_dir = getattr(cfg, "checkpoint_dir", "./core_checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)

            ckpt_path = os.path.join(
                ckpt_dir,
                getattr(cfg, "bmaml_ckpt_name", f"{cfg.dataset_source}_bmaml_best_re.pt"),
            )

            export_state = {
                "model_state": best_state["model"],
                "meta_thetas": [t.detach().cpu() for t in best_state["meta_thetas"]],
                "config": best_state.get("cfg", cfg).__dict__,
                "seq_scaler": seq_scaler,
                "sum_scaler": sum_scaler,
                "max_rul_train": max_rul_train,
                "feature_cols": feature_cols,
                "target_col": target_col,
                "train_bids": train_bids,
                "val_bids": val_bids,
                "test_bids": test_bids,
            }

            torch.save(export_state, ckpt_path)
            print(f"[SAVE] Epoch {epoch:03d} best checkpoint saved to {ckpt_path}")

            # wandb artifact도 매번 올리고 싶으면 여기로 이동
            if use_wandb and wandb_run is not None:
                art = wandb.Artifact(name=f"{run_name}_best", type="model")
                art.add_file(ckpt_path)
                wandb_run.log_artifact(art)

        else:
            patience += 1
            if patience >= cfg.max_patience:
                print("[Early Stop] patience reached.")
                break


    # 7) Best state 로드 후 meta-test
    # 6-1) 베스트 체크포인트 저장 (배포 아티팩트 포함)
    if best_state is not None:
        ckpt_dir = getattr(cfg, "checkpoint_dir", "./core_checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)

        ckpt_path = os.path.join(
            ckpt_dir,
            getattr(cfg, "bmaml_ckpt_name", f"{cfg.dataset_source}_bmaml_best_re_.pt"),
        )

        export_state = {
            "model_state": best_state["model"],
            "meta_thetas": [t.detach().cpu() for t in best_state["meta_thetas"]],
            "config": best_state.get("cfg", cfg).__dict__,
            "seq_scaler": seq_scaler,
            "sum_scaler": sum_scaler,
            "max_rul_train": max_rul_train,
            "feature_cols": feature_cols,
            "target_col": target_col,
            "train_bids": train_bids,
            "val_bids": val_bids,
            "test_bids": test_bids,
        }

        torch.save(export_state, ckpt_path)
        print(f"[SAVE] Best checkpoint saved to {ckpt_path}")

        if use_wandb and wandb_run is not None and run_name is not None:
            art = wandb.Artifact(name=f"{run_name}_best", type="model")
            art.add_file(ckpt_path)
            wandb_run.log_artifact(art)

        # 테스트 전에 베스트 state 로 모델/파티클 로드
        model.load_state_dict(best_state["model"])
        meta_thetas = best_state["meta_thetas"]

    # 7) Held-out 배터리로 meta-test
    rmse_t, r2_t, mae_t, mse_t, calib_t, aux_t = meta_evaluate(
        model, vecizer, meta_thetas, test_loader, cfg, max_rul_train
    )
    print(
        f"[FINAL TEST] RMSE={rmse_t:.3f} | MAE={mae_t:.3f} | R2={r2_t:.3f} | "
        f"MSE={mse_t:.4f} | CAL={calib_t:.4f} | AUX={aux_t:.4f}"
    )

    if use_wandb and wandb_run is not None:
        wandb_run.log(
            {
                "test_rmse": rmse_t,
                "test_mae": mae_t,
                "test_r2": r2_t,
                "test_mse": mse_t,
                "test_calib": calib_t,
                "test_aux": aux_t,
            }
        )
        wandb_run.finish()

if __name__ == "__main__":
    main()


# # 1단계: backbone supervised 학습
# python -m deep_learning.core.train_backbone

# # 2단계: BMAML-SVGD meta 학습 (기존대로)
# python -m deep_learning.core.train_meta

