# deep_learning/core/train_backbone.py
# CEEMDAN-Transformer-DNN 백본을 순수 supervised로 먼저 학습시키는 엔트리포인트.
# - BMAML-SVGD 메타 학습 전에 한 번 돌려서 초기 가중치 품질을 끌어올리는 용도.
# - 출력: backbone 전용 체크포인트 (model_state + scaler 등)

import os
import random
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from deep_learning.core.config import Config
from deep_learning.core.meta_db_loader import load_cycle_db_for_meta
from deep_learning.core.data_seq_group import group_data_by_battery_from_df
from deep_learning.core.scalers import CustomRobustScaler3D, scale_rul_array, unscale_rul_array
from deep_learning.core.models import MultiTaskRULModel
from deep_learning.core.meta_utils import DEVICE

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


def fit_scalers(
    grouped: dict, train_bids: List[str], c: Config
) -> Tuple[CustomRobustScaler3D, RobustScaler, float]:
    """
    meta 쪽과 동일한 로직으로 seq / summary scaler 및 max_rul_train 계산.
    """
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

    # NASA + CACLE 통합일 경우를 대비해서 train 배터리 기준 max_rul 계산
    max_rul_train = max(float(grouped[b]["max_rul"]) for b in train_bids)
    return seq_scaler, sum_scaler, max_rul_train


class SupervisedSeqDataset(Dataset):
    """
    grouped_data 를 이용한 순수 supervised Dataset.
    - 각 (battery, window) 를 하나의 샘플로 사용.
    - 입력: scaled seq, sum
    - 타겟: scaled RUL
    """

    def __init__(
        self,
        grouped_data: dict,
        battery_ids: List[str],
        c: Config,
        seq_scaler: CustomRobustScaler3D,
        sum_scaler: RobustScaler,
        max_rul_train: float,
    ):
        self.samples = []

        for bid in battery_ids:
            d = grouped_data[bid]
            # 이미 group_data_by_battery_from_df에서 drop_zero_tail 적용됨
            seq = seq_scaler.transform(d["seq"])        # (N, T, F)
            summ = sum_scaler.transform(d["sum"])       # (N, D)
            rul_scaled = scale_rul_array(
                d["rul"].astype(np.float32),
                c.rul_mode,
                max_rul_train,
            )                                           # (N,)

            for i in range(len(rul_scaled)):
                self.samples.append(
                    (
                        seq[i],                 # (T, F)
                        summ[i],                # (D,)
                        float(rul_scaled[i]),   # scalar
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, summ, rul = self.samples[idx]
        return (
            torch.from_numpy(seq).float(),
            torch.from_numpy(summ).float(),
            torch.tensor(rul, dtype=torch.float32),
        )

# ---------------------------------------------------------------------
# Train / Eval Loop
# ---------------------------------------------------------------------

def train_epoch(model, loader, opt, c: Config):
    model.train()
    total_loss = 0.0
    n_samples = 0

    mse = torch.nn.MSELoss()
    huber = torch.nn.SmoothL1Loss(beta=0.1)  # 작은 에러는 L2, 큰 에러는 L1처럼

    for seq, summ, rul in loader:
        seq = torch.nan_to_num(seq.to(DEVICE), nan=0.0, posinf=0.0, neginf=0.0)
        summ = torch.nan_to_num(summ.to(DEVICE), nan=0.0, posinf=0.0, neginf=0.0)
        rul = torch.nan_to_num(rul.to(DEVICE), nan=0.0, posinf=0.0, neginf=0.0)

        opt.zero_grad(set_to_none=True)

        out, _, _, _ = model(seq, summ)
        out = torch.nan_to_num(out.squeeze(-1), nan=0.0, posinf=0.0, neginf=0.0)

        # 두 손실을 섞어서 사용
        loss_mse = mse(out, rul)
        loss_huber = huber(out, rul)
        loss = 0.5 * loss_mse + 0.5 * loss_huber

        loss.backward()

        clip_val = float(getattr(c, "outer_clip", 1.0))
        clip_grad_norm_(model.parameters(), clip_val)

        opt.step()

        bs = seq.size(0)
        total_loss += float(loss.item()) * bs
        n_samples += bs

    return total_loss / max(1, n_samples)


def evaluate(model, loader, c: Config, max_rul_train: float):
    model.eval()
    all_y_scaled, all_pred_scaled = [], []

    with torch.no_grad():
        for seq, summ, rul in loader:
            seq = torch.nan_to_num(seq.to(DEVICE), nan=0.0, posinf=0.0, neginf=0.0)
            summ = torch.nan_to_num(summ.to(DEVICE), nan=0.0, posinf=0.0, neginf=0.0)
            rul = torch.nan_to_num(rul.to(DEVICE), nan=0.0, posinf=0.0, neginf=0.0)

            out, _, _, _ = model(seq, summ)
            out = torch.nan_to_num(out.squeeze(-1), nan=0.0, posinf=0.0, neginf=0.0)

            all_y_scaled.append(rul.detach().cpu().numpy())
            all_pred_scaled.append(out.detach().cpu().numpy())

    if not all_y_scaled:
        return 1e6, -1e6, 1e6, 1e6

    y_scaled = np.concatenate(all_y_scaled)
    pred_scaled = np.concatenate(all_pred_scaled)

    # 스케일 해제 후 실제 RUL 단위로 metric 계산
    y = unscale_rul_array(y_scaled, c.rul_mode, max_rul_train)
    pred = unscale_rul_array(pred_scaled, c.rul_mode, max_rul_train)

    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    mae = float(mean_absolute_error(y, pred))
    if len(y) > 1 and np.var(y) > 1e-8:
        r2 = float(r2_score(y, pred))
        r2 = float(np.clip(r2, -10.0, 1.0))
    else:
        r2 = 0.0

    # 참고용: RUL > 0 subset에서의 RMSE (tail 0부분 제외한 성능이 궁금할 때)
    mask = y > 0.0
    if mask.sum() > 0:
        rmse_pos = float(np.sqrt(mean_squared_error(y[mask], pred[mask])))
    else:
        rmse_pos = float("nan")

    return rmse, r2, mae, rmse_pos

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    cfg = Config()
    set_seed(cfg.RANDOM_SEED)

    # 1) 통합 cycle-level DB 로드 (NASA / CACLE + CEEMDAN 설정 등)
    df, target_col, feature_cols, bid_col, cyc_col = load_cycle_db_for_meta(cfg)
    
    # 2) 배터리 단위 그룹핑 (seq / summary / rul)
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
        raise RuntimeError(f"Too few batteries ({len(all_bids)}) for backbone training.")

    rng = np.random.RandomState(cfg.RANDOM_SEED)
    rng.shuffle(all_bids)

    n_total = len(all_bids)
    n_train = max(1, int(n_total * 0.6))
    n_val = max(1, int(n_total * 0.2))
    n_test = max(1, n_total - n_train - n_val)

    train_bids = all_bids[:n_train]
    val_bids = all_bids[n_train: n_train + n_val]
    test_bids = all_bids[n_train + n_val: n_train + n_val + n_test]

    print(f"[BACKBONE SPLIT] train={len(train_bids)}, val={len(val_bids)}, test={len(test_bids)}")
    print("[BACKBONE TRAIN BIDS]", train_bids)
    print("[BACKBONE VAL   BIDS]", val_bids)
    print("[BACKBONE TEST  BIDS]", test_bids)

    # 3) Scaler fit (train 배터리 기준)
    seq_scaler, sum_scaler, max_rul_train = fit_scalers(grouped, train_bids, cfg)
    print(f"[BACKBONE] max_rul_train={max_rul_train:.3f}")

    # 4) Dataset / DataLoader
    train_ds = SupervisedSeqDataset(grouped, train_bids, cfg, seq_scaler, sum_scaler, max_rul_train)
    val_ds = SupervisedSeqDataset(grouped, val_bids, cfg, seq_scaler, sum_scaler, max_rul_train)
    test_ds = SupervisedSeqDataset(grouped, test_bids, cfg, seq_scaler, sum_scaler, max_rul_train)

    train_loader = DataLoader(
        train_ds,
        batch_size=getattr(cfg, "supervised_batch_size", 64),
        shuffle=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=getattr(cfg, "supervised_batch_size", 64),
        shuffle=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=getattr(cfg, "supervised_batch_size", 64),
        shuffle=False,
    )

    # 5) 모델 / 옵티마이저 초기화
    sd = grouped[train_bids[0]]["seq"].shape[-1]
    sm = grouped[train_bids[0]]["sum"].shape[-1]
    model = MultiTaskRULModel(sd, sm, cfg).to(DEVICE)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # 6) 학습 루프 (단순 early stopping; val RMSE 기준)
    best_rmse = 1e9
    best_state_dict = None
    patience = 0
    max_patience = int(getattr(cfg, "max_patience_backbone", cfg.max_patience))

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_epoch(model, train_loader, opt, cfg)
        rmse, r2, mae, rmse_pos = evaluate(model, val_loader, cfg, max_rul_train)
        rmse_norm = rmse / (max_rul_train + 1e-8)

        print(
            f"[BACKBONE Epoch {epoch:03d}] "
            f"TrainLoss={train_loss:.4f} | "
            f"RMSE={rmse:.3f} ({rmse_norm:.3f}x maxRUL) | "
            f"RMSE(RUL>0)={rmse_pos:.3f} | "
            f"MAE={mae:.3f} | R2={r2:.3f}"
        )

        if rmse < best_rmse:
            best_rmse = rmse
            patience = 0
            best_state_dict = model.state_dict()
        else:
            patience += 1
            if patience >= max_patience:
                print("[BACKBONE Early Stop] patience reached.")
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # 최종 test metric
    rmse_t, r2_t, mae_t, rmse_pos_t = evaluate(model, test_loader, cfg, max_rul_train)
    rmse_norm_t = rmse_t / (max_rul_train + 1e-8)
    print(
        f"[BACKBONE FINAL TEST] RMSE={rmse_t:.3f} ({rmse_norm_t:.3f}x maxRUL) | "
        f"RMSE(RUL>0)={rmse_pos_t:.3f} | "
        f"MAE={mae_t:.3f} | R2={r2_t:.3f}"
    )

    # 7) 체크포인트 저장 (BMAML meta 초기화용)
    ckpt_dir = getattr(cfg, "checkpoint_dir", "./core_checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    backbone_ckpt_path = os.path.join(
    ckpt_dir,
        f"{cfg.dataset_source}_backbone_ceemdan_trf_dnn_re_(2).pt",
    )

    export_state = {
        "model_state": model.state_dict(),
        "config": cfg.__dict__,
        "seq_scaler": seq_scaler,
        "sum_scaler": sum_scaler,
        "max_rul_train": max_rul_train,
        "feature_cols": feature_cols,
        "target_col": target_col,
        "train_bids": train_bids,
        "val_bids": val_bids,
        "test_bids": test_bids,
    }
    torch.save(export_state, backbone_ckpt_path)
    print(f"[BACKBONE SAVE] checkpoint saved to {backbone_ckpt_path}")


if __name__ == "__main__":
    main()
