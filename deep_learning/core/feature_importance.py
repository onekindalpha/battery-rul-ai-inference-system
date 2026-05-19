import os
import random
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

import torch
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error

from deep_learning.core.config import Config
from deep_learning.core.meta_db_loader import load_cycle_db_for_meta
from deep_learning.core.data_seq_group import group_data_by_battery_from_df
from deep_learning.core.scalers import (
    CustomRobustScaler3D,
    scale_rul_array,
    unscale_rul_array,
)
from deep_learning.core.models import MultiTaskRULModel
from deep_learning.core.meta_utils import DEVICE


# ---------------------------------------------------------
# Utils
# ---------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_scalers_for_bids(
    grouped: dict, train_bids: List[str], c: Config
) -> Tuple[CustomRobustScaler3D, RobustScaler, float]:
    """
    train_backbone.py 의 fit_scalers 로직을 그대로 복붙한 버전.
    (import 순환 피하기 위해 여기 다시 정의)
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

    seq_scaler = CustomRobustScaler3D(
        p=90, min_iqr=1e-3, clip_value=c.seq_clip
    )
    seq_scaler.fit(seq_arr)

    sum_scaler = RobustScaler()
    sum_scaler.fit(sum_arr)

    # train 배터리 기준 max_rul
    max_rul_train = max(float(grouped[b]["max_rul"]) for b in train_bids)
    return seq_scaler, sum_scaler, max_rul_train


def build_eval_arrays(
    grouped: dict,
    eval_bids: List[str],
    c: Config,
    seq_scaler: CustomRobustScaler3D,
    sum_scaler: RobustScaler,
    max_rul_train: float,
):
    """
    val + test 배터리에서 전체 윈도우를 모아서
    모델에 바로 넣을 수 있는 numpy 배열로 만든다.
    """
    seq_list, sum_list, rul_scaled_list = [], [], []

    for bid in eval_bids:
        d = grouped[bid]
        seq = seq_scaler.transform(d["seq"])      # (N, T, F)
        summ = sum_scaler.transform(d["sum"])     # (N, D)
        rul_scaled = scale_rul_array(
            d["rul"].astype(np.float32),
            c.rul_mode,
            max_rul_train,
        )                                         # (N,)

        seq_list.append(seq)
        sum_list.append(summ)
        rul_scaled_list.append(rul_scaled)

    if not seq_list:
        raise RuntimeError("No eval samples constructed (check bids / grouped).")

    X_seq = np.concatenate(seq_list, axis=0)
    X_sum = np.concatenate(sum_list, axis=0)
    y_scaled = np.concatenate(rul_scaled_list, axis=0)

    return X_seq, X_sum, y_scaled


def eval_rmse(
    model: torch.nn.Module,
    X_seq: np.ndarray,
    X_sum: np.ndarray,
    y_scaled: np.ndarray,
    c: Config,
    max_rul_train: float,
    batch_size: int = 256,
) -> float:
    """
    현재 입력 배열(X_seq, X_sum, y_scaled)에 대해
    실제 RUL 단위의 RMSE를 계산.
    """
    model.eval()
    n = X_seq.shape[0]
    preds_scaled = []

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            seq_batch = torch.from_numpy(X_seq[start:end]).float().to(DEVICE)
            sum_batch = torch.from_numpy(X_sum[start:end]).float().to(DEVICE)

            seq_batch = torch.nan_to_num(seq_batch, nan=0.0, posinf=0.0, neginf=0.0)
            sum_batch = torch.nan_to_num(sum_batch, nan=0.0, posinf=0.0, neginf=0.0)

            out, _, _, _ = model(seq_batch, sum_batch)
            out = torch.nan_to_num(out.squeeze(-1), nan=0.0, posinf=0.0, neginf=0.0)

            preds_scaled.append(out.detach().cpu().numpy())

    if not preds_scaled:
        return float("inf")

    pred_scaled = np.concatenate(preds_scaled, axis=0)

    # 스케일 해제
    y = unscale_rul_array(y_scaled, c.rul_mode, max_rul_train)
    pred = unscale_rul_array(pred_scaled, c.rul_mode, max_rul_train)

    rmse = float(np.sqrt(mean_squared_error(y, pred)))
    return rmse


def compute_seq_permutation_importance(
    model: torch.nn.Module,
    X_seq: np.ndarray,
    X_sum: np.ndarray,
    y_scaled: np.ndarray,
    c: Config,
    max_rul_train: float,
    feature_names: List[str],
    random_state: int = 42,
) -> List[Tuple[str, float]]:
    """
    시퀀스 브랜치(CEEMDAN + 기본 feature) 기준 permutation importance.
    - RMSE 증가량을 importance로 사용.
    - feature_names 길이 == X_seq.shape[-1] 이어야 함.
    """
    assert X_seq.shape[-1] == len(feature_names), \
        f"X_seq last dim {X_seq.shape[-1]} vs feature_names {len(feature_names)}"

    rng = np.random.RandomState(random_state)

    # baseline RMSE
    base_rmse = eval_rmse(model, X_seq, X_sum, y_scaled, c, max_rul_train)
    print(f"[Base] RMSE = {base_rmse:.4f}")

    N, T, F = X_seq.shape
    importances: List[Tuple[str, float]] = []

    for j, name in enumerate(feature_names):
        X_perm = X_seq.copy()

        # (N, T) flatten → shuffle → reshape
        flat = X_perm[:, :, j].reshape(-1)
        rng.shuffle(flat)
        X_perm[:, :, j] = flat.reshape(N, T)

        rmse_perm = eval_rmse(model, X_perm, X_sum, y_scaled, c, max_rul_train)
        delta = rmse_perm - base_rmse
        importances.append((name, float(delta)))

        print(f"[Perm] {j:03d} {name:30s}  RMSE={rmse_perm:.4f}  Δ={delta:.4f}")

    # Δ RMSE 큰 순으로 정렬 (내림차순)
    importances.sort(key=lambda x: x[1], reverse=True)
    return importances


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():
    cfg = Config()
    set_seed(cfg.RANDOM_SEED)

    # 여기서 dataset_source, use_ceemdan, feature_set 등은
    # config.py에서 수정해두면 그대로 따라감.
    print(f"[CFG] dataset_source={cfg.dataset_source}, db_type={cfg.db_type}, use_ceemdan={cfg.use_ceemdan}")

    # 1) cycle-level DB 로딩
    df, target_col, feature_cols, bid_col, cyc_col = load_cycle_db_for_meta(cfg)
    print(f"[DB] rows={len(df)}, target={target_col}, #features={len(feature_cols)}")

    # seq branch feature 이름 (feature_cols + IMF 계열) 구성
    imf_cols = [c for c in df.columns if c.startswith("IMF")]
    tf_cols = sorted(set(list(feature_cols) + imf_cols))
    print(f"[Seq Features] F={len(tf_cols)}")

    # 2) 배터리별 윈도우 그룹핑
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
        raise RuntimeError(f"Too few batteries ({len(all_bids)}) for feature importance.")

    rng = np.random.RandomState(cfg.RANDOM_SEED)
    rng.shuffle(all_bids)

    n_total = len(all_bids)
    n_train = max(1, int(n_total * 0.6))
    n_val = max(1, int(n_total * 0.2))
    n_test = max(1, n_total - n_train - n_val)

    train_bids = all_bids[:n_train]
    val_bids = all_bids[n_train: n_train + n_val]
    test_bids = all_bids[n_train + n_val: n_train + n_val + n_test]

    print(f"[SPLIT] train={len(train_bids)}, val={len(val_bids)}, test={len(test_bids)}")

    # 3) scaler fit (train 기준)
    seq_scaler, sum_scaler, max_rul_train = fit_scalers_for_bids(
        grouped, train_bids, cfg
    )

    # 4) eval 배열 구성 (val + test)
    eval_bids = val_bids + test_bids
    X_seq, X_sum, y_scaled = build_eval_arrays(
        grouped,
        eval_bids,
        cfg,
        seq_scaler,
        sum_scaler,
        max_rul_train,
    )
    print(f"[Eval Arrays] seq={X_seq.shape}, sum={X_sum.shape}, y={y_scaled.shape}")

    BACKBONE_CKPT = "/Users/velocitygoal/Desktop/battery_project/v11/core_checkpoints/nasa_backbone_ceemdan_trf_dnn.pt"

    # 5) 모델 + 백본 checkpoint 로드
    sd = X_seq.shape[-1]   # seq feature dim (with IMF)
    sm = X_sum.shape[-1]   # summary feature dim

    model = MultiTaskRULModel(sd, sm, cfg).to(DEVICE)
    backbone_ckpt_path = BACKBONE_CKPT  # <<< 하드코딩
    
    if backbone_ckpt_path and os.path.exists(backbone_ckpt_path):
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
    else:
        print(f"[WARN] No backbone checkpoint found at {backbone_ckpt_path}, using random init.")
    
    # 6) Permutation importance 계산
    importances = compute_seq_permutation_importance(
        model=model,
        X_seq=X_seq,
        X_sum=X_sum,
        y_scaled=y_scaled,
        c=cfg,
        max_rul_train=max_rul_train,
        feature_names=tf_cols,
        random_state=cfg.RANDOM_SEED,
    )

    print("\n=== Permutation Feature Importance (Seq branch, ΔRMSE desc) ===")
    for name, delta in importances:
        print(f"{name:30s}  ΔRMSE={delta:.4f}")
    

    # numpy/정렬용 배열로 변환
    names = [n for n, d in importances]
    deltas = np.array([d for n, d in importances])

    # 중요도 절대값 기준으로 정렬하면 보기 편함
    order = np.argsort(-np.abs(deltas))   # 내림차순
    names_sorted = [names[i] for i in order]
    deltas_sorted = deltas[order]

    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(names_sorted))

    plt.barh(y_pos, deltas_sorted)  # 양수 = 도움, 음수 = 해로움
    plt.yticks(y_pos, names_sorted)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Δ RMSE (perm - base)")
    plt.title("Permutation Feature Importance (Backbone, Seq branch)")

    plt.tight_layout()
    plt.savefig("backbone_perm_importance.png", dpi=200)

if __name__ == "__main__":
    main()
