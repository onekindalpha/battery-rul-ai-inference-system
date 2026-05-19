# deep_learning/core/feature_shap_bmaml.py
"""
BMAML 체크포인트(cacle_bmaml_best_re_.pt 등)의 모델을 기준으로
seq-branch 40개 피처에 대한 SHAP 중요도를 계산하는 스크립트.

아이디어:
- BMAML checkpoint 안의 model_state 를 하나의 고정 RUL 모델로 보고 사용 (inner-loop 적응은 여기서 제외).
- cycle-level DB (Config.dataset_source: nasa / cacle / both) 로부터 seq 데이터를 모으고, seq_scaler 로 스케일링.
- 각 샘플의 seq: [T, F=40] 을 평탄화 → [T*F] 로 만들고, SHAP KernelExplainer 를 사용.
- 계산된 SHAP 값은 time-axis(T) 방향으로 합쳐서, feature 별 importance 를 얻는다.

성능(시간) 줄이기용 옵션:
- --background: SHAP background 샘플 수 (작게 줄일수록 빠름)
- --n_explain: 실제로 SHAP을 계산할 샘플 수 (작게 줄일수록 빠름)
- --nsamples: KernelExplainer 의 샘플 수 (작게 줄일수록 빠름)
- --keep_last_t: 마지막 T step만 사용해서 time 축 길이 줄이기 (T 줄이면 차원 ↓ → 속도 많이 빨라짐)
- --outdir: 결과 png/json 저장 디렉토리

사용 예시:
    python -m deep_learning.core.feature_shap_bmaml \\
        --ckpt /path/to/nasa_bmaml_best.pt \\
        --background 30 \\
        --n_explain 20 \\
        --nsamples 50 \\
        --keep_last_t 30 \\
        --outdir /path/to/output_dir
"""

import os
import argparse
import random
from typing import List, Optional

import numpy as np
import torch
from sklearn.preprocessing import RobustScaler

import shap
import matplotlib.pyplot as plt

from deep_learning.core.config import Config
from deep_learning.core.meta_db_loader import load_cycle_db_for_meta
from deep_learning.core.data_seq_group import group_data_by_battery_from_df
from deep_learning.core.scalers import CustomRobustScaler3D, unscale_rul_array
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


def fit_scalers_for_seq_sum(grouped, train_bids: List[str], cfg: Config):
    """
    meta / backbone 쪽에서 쓰던 로직과 동일한 컨셉으로
    seq_scaler(3D) + sum_scaler(2D) + max_rul_train 을 만든다.
    """
    seq_list, sum_list = [], []

    for bid in train_bids:
        data = grouped[bid]
        seq = data["seq"]       # (N_i, T, F)
        summ = data["sum"]      # (N_i, S)

        n = len(seq)
        fit_len = int(n * cfg.scaler_fit_ratio)
        fit_len = max(fit_len, cfg.k_shot + cfg.q_query + 1)
        fit_len = min(fit_len, n)

        seq_list.append(seq[:fit_len])
        sum_list.append(summ[:fit_len])

    seq_arr = np.concatenate(seq_list, axis=0)  # (N_fit, T, F)
    sum_arr = np.concatenate(sum_list, axis=0)  # (N_fit, S)

    seq_scaler = CustomRobustScaler3D(p=90, min_iqr=1e-3, clip_value=cfg.seq_clip)
    seq_scaler.fit(seq_arr)

    sum_scaler = RobustScaler()
    sum_scaler.fit(sum_arr)

    max_rul_train = max(float(grouped[b]["max_rul"]) for b in train_bids)
    return seq_scaler, sum_scaler, max_rul_train


def build_flat_seq_dataset(
    grouped,
    bid_list,
    seq_scaler,
    sum_scaler,
    keep_last_t: Optional[int] = None,
):
    """
    grouped dict 에서 선택한 배터리들만 모아서:
    - seq_scaled: (N, T, F)
    - sum_scaled: (N, S)
    - seq_flat : (N, T*F)  ← SHAP input
      * keep_last_t 가 설정되면, 마지막 keep_last_t step만 사용하여 T를 줄인다.
    - some_sum : (S,)      ← summary feature는 여기서는 전체 평균 한 개만 사용
    """
    seq_arr_list, sum_arr_list = [], []

    for bid in bid_list:
        g = grouped[bid]
        seq = g["seq"]   # (N_i, T, F)
        summ = g["sum"]  # (N_i, S)

        seq_arr_list.append(seq)
        sum_arr_list.append(summ)

    seq_arr = np.concatenate(seq_arr_list, axis=0)  # (N, T, F)
    sum_arr = np.concatenate(sum_arr_list, axis=0)  # (N, S)

    # NaN 등 방지
    seq_arr = np.nan_to_num(seq_arr, nan=0.0, posinf=0.0, neginf=0.0)
    sum_arr = np.nan_to_num(sum_arr, nan=0.0, posinf=0.0, neginf=0.0)

    # 스케일링
    seq_sc = seq_scaler.transform(seq_arr)  # (N, T, F)
    sum_sc = sum_scaler.transform(sum_arr)  # (N, S)

    # time 축 줄이기: 마지막 keep_last_t step만 사용
    if keep_last_t is not None and keep_last_t > 0 and keep_last_t < seq_sc.shape[1]:
        seq_sc = seq_sc[:, -keep_last_t:, :]  # (N, keep_last_t, F)

    N, T, F = seq_sc.shape
    seq_flat = seq_sc.reshape(N, T * F)

    # summary는 여기서는 한 개의 대표값(평균)만 사용해서,
    # 모든 SHAP 샘플에서 고정된 입력으로 넣는다.
    some_sum = sum_sc.mean(axis=0)  # (S,)

    return seq_flat, T, F, some_sum


# ---------------------------------------------------------------------
# SHAP main logic
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="BMAML checkpoint path (e.g., nasa_bmaml_best.pt)",
    )
    parser.add_argument(
        "--background",
        type=int,
        default=30,  # 원래 100 → 기본값 낮춰서 속도 개선
        help="SHAP background sample size (KernelExplainer)",
    )
    parser.add_argument(
        "--n_explain",
        type=int,
        default=20,
        help="Number of samples to explain with SHAP (X_explain size)",
    )
    parser.add_argument(
        "--nsamples",
        type=int,
        default=50,  # 원래 100 → 기본값 낮춰서 속도 개선
        help="SHAP nsamples (KernelExplainer complexity 조절)",
    )
    parser.add_argument(
        "--keep_last_t",
        type=int,
        default=None,
        help="If set, only use last T timesteps for SHAP (reduces dimension).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=".",
        help="Directory to save SHAP outputs (png, json).",
    )
    args = parser.parse_args()

    set_seed(42)

    ckpt_path = args.ckpt
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    print(f"[LOAD] BMAML checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # -----------------------------------------------------------------
    # 1) config 로드 (checkpoint 기준)
    # -----------------------------------------------------------------
    cfg_dict = ckpt.get("config", {})
    cfg = Config()
    for k, v in cfg_dict.items():
        setattr(cfg, k, v)

    print(f"[CFG] dataset_source={cfg.dataset_source}, db_type={cfg.db_type}, use_ceemdan={cfg.use_ceemdan}")

    # -----------------------------------------------------------------
    # 2) 데이터 로드 + grouped
    # -----------------------------------------------------------------
    df, target_col, feature_cols, bid_col, cyc_col = load_cycle_db_for_meta(cfg)
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
        raise RuntimeError(f"Too few batteries for SHAP: {len(all_bids)}")

    print(f"[DB] bids={all_bids}")
    print(f"[DB] feature_cols ({len(feature_cols)}): {feature_cols}")

    # train/val/test split 재현 (없으면 랜덤)
    if all(k in ckpt for k in ("train_bids", "val_bids", "test_bids")):
        train_bids = [b for b in ckpt["train_bids"] if b in all_bids]
        val_bids   = [b for b in ckpt["val_bids"]   if b in all_bids]
        test_bids  = [b for b in ckpt["test_bids"]  if b in all_bids]
    else:
        rng = np.random.RandomState(cfg.RANDOM_SEED)
        rng.shuffle(all_bids)
        n_total = len(all_bids)
        n_train = max(1, int(n_total * 0.6))
        n_val   = max(1, int(n_total * 0.2))
        train_bids = all_bids[:n_train]
        val_bids   = all_bids[n_train : n_train + n_val]
        test_bids  = all_bids[n_train + n_val :]

    print(f"[SPLIT] train={train_bids}, val={val_bids}, test={test_bids}")

    # -----------------------------------------------------------------
    # 3) scaler / max_rul_train 로드 (ckpt에 있으면 그거 사용)
    # -----------------------------------------------------------------
    if all(k in ckpt for k in ("seq_scaler", "sum_scaler", "max_rul_train")):
        print("[INIT] Using scalers from checkpoint.")
        seq_scaler = ckpt["seq_scaler"]
        sum_scaler = ckpt["sum_scaler"]
        max_rul_train = ckpt["max_rul_train"]
    else:
        print("[INIT] No scalers in checkpoint, fitting from train_bids.")
        seq_scaler, sum_scaler, max_rul_train = fit_scalers_for_seq_sum(
            grouped, train_bids, cfg
        )

    # -----------------------------------------------------------------
    # 4) 모델 로드 (BMAML model_state를 고정된 RUL 모델로 사용)
    # -----------------------------------------------------------------
    first_bid = train_bids[0]
    sd = grouped[first_bid]["seq"].shape[-1]  # F=40
    sm = grouped[first_bid]["sum"].shape[-1]  # summary dimension

    model = MultiTaskRULModel(sd, sm, cfg).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"[MODEL] sd={sd}, sm={sm}")

    # -----------------------------------------------------------------
    # 5) SHAP용 데이터 구성 (여기서는 test_bids 기준으로)
    # -----------------------------------------------------------------
    if len(test_bids) == 0:
        # fallback
        use_bids = val_bids if len(val_bids) > 0 else train_bids
    else:
        use_bids = test_bids

    print(f"[SHAP] using bids for SHAP: {use_bids}")

    seq_flat, T, F, some_sum = build_flat_seq_dataset(
        grouped, use_bids, seq_scaler, sum_scaler, keep_last_t=args.keep_last_t
    )
    N, D = seq_flat.shape
    print(f"[SHAP] seq_flat shape = {seq_flat.shape}, T={T}, F={F}, keep_last_t={args.keep_last_t}")

    # background & explain sample 추출
    rng = np.random.RandomState(0)
    bg_size = min(args.background, N)
    bg_idx = rng.choice(N, size=bg_size, replace=False)
    background = seq_flat[bg_idx]

    exp_size = min(args.n_explain, N)  # background와 분리
    exp_idx = rng.choice(N, size=exp_size, replace=False)
    X_explain = seq_flat[exp_idx]

    print(f"[SHAP] background size = {background.shape[0]}, explain size = {X_explain.shape[0]}")

    # summary 입력은 고정값 some_sum 사용
    some_sum_torch = torch.from_numpy(some_sum).float().to(DEVICE).unsqueeze(0)  # (1, S)

    # -----------------------------------------------------------------
    # 6) 예측 함수 정의: X_flat -> RUL (실제 스케일)
    # -----------------------------------------------------------------
    def predict_fn(X_flat_np: np.ndarray) -> np.ndarray:
        """
        X_flat_np: (n_samples, T*F)  (이미 seq_scaler 로 스케일된 상태)
        """
        n_samp = X_flat_np.shape[0]
        seq_sc = X_flat_np.reshape(n_samp, T, F)  # (N, T, F)

        seq_t = torch.from_numpy(seq_sc).float().to(DEVICE)

        # summary 는 모두 같은 값 some_sum 을 사용
        sum_rep = some_sum_torch.repeat(n_samp, 1)  # (N, S)

        with torch.no_grad():
            out, _, _, _ = model(seq_t, sum_rep)
            out_sc = out.squeeze(-1).detach().cpu().numpy()  # scaled RUL

        # 스케일 복원
        out_real = unscale_rul_array(out_sc, cfg.rul_mode, max_rul_train)
        return out_real

    # -----------------------------------------------------------------
    # 7) SHAP KernelExplainer 실행
    # -----------------------------------------------------------------
    print(
        f"[SHAP] Building KernelExplainer "
        f"(background={background.shape[0]}, n_explain={X_explain.shape[0]}, nsamples={args.nsamples})"
    )
    explainer = shap.KernelExplainer(predict_fn, background)

    print("[SHAP] Computing shap_values ... (this may take some time)")
    shap_values = explainer.shap_values(X_explain, nsamples=args.nsamples)
    # shap_values: (N_explain, D)

    shap_values = np.array(shap_values)  # ensure ndarray
    # 평균 절댓값으로 feature 중요도 계산
    shap_abs_mean = np.mean(np.abs(shap_values), axis=0)  # (D,)

    # time-axis(T) 로 합쳐서 feature 단위(F) 중요도 계산
    shap_per_feature = shap_abs_mean.reshape(T, F).sum(axis=0)  # (F,)

    # feature_cols 가 seq feature 순서와 동일하다고 가정 (load_cycle_db_for_meta 에서 사용)
    if len(feature_cols) != F:
        print(
            f"[WARN] feature_cols len ({len(feature_cols)}) != F ({F}), "
            f"이 경우 feature 이름 매칭이 어긋날 수 있습니다."
        )
        feature_names = [f"f_{i}" for i in range(F)]
    else:
        feature_names = feature_cols

    # 정렬 및 출력
    order = np.argsort(-shap_per_feature)  # 큰 값부터
    print("\n=== SHAP Feature Importance (Seq branch, |φ| aggregated over time) ===")
    for idx in order:
        print(f"{feature_names[idx]:30s}  mean|φ|={shap_per_feature[idx]:.6f}")

    # -----------------------------------------------------------------
    # 8) 시각화 (barh plot) + 저장
    # -----------------------------------------------------------------
    names_sorted = [feature_names[i] for i in order]
    vals_sorted = shap_per_feature[order]

    plt.figure(figsize=(10, 8))
    y_pos = np.arange(len(names_sorted))

    plt.barh(y_pos, vals_sorted)
    plt.yticks(y_pos, names_sorted)
    plt.xlabel("Mean |SHAP value| (aggregated over time)")
    plt.title("BMAML Seq Feature Importance (SHAP, nasa_bmaml_best)")

    plt.gca().invert_yaxis()  # 상단이 가장 큰 값
    plt.tight_layout()

    out_path = os.path.join(outdir, "bmaml_shap_seq_feature_importance.png")
    plt.savefig(out_path, dpi=200)
    print(f"[SAVE] Plot saved to {out_path}")

    # 👉 JSON으로도 저장 (대시보드용)
    shap_dict = {
        "feature_names": names_sorted,
        "importance": vals_sorted.tolist(),
    }
    json_out = os.path.join(outdir, "bmaml_shap_seq_feature_importance.json")
    import json
    with open(json_out, "w") as f:
        json.dump(shap_dict, f, indent=2)
    print(f"[SAVE] SHAP importance JSON saved to {json_out}")


if __name__ == "__main__":
    main()

# 간단한 버전
# python -m deep_learning.core.feature_shap_bmaml \
#   --ckpt "/Users/velocitygoal/Desktop/battery_project/v11/deep_learning/core/core_checkpoints/cacle_bmaml_best_re_.pt" \
#   --background 30 \
#   --n_explain 20 \
#   --nsamples 50 \
#   --keep_last_t 30 \
#   --outdir "/Users/velocitygoal/Desktop/battery_project/v11/shap_outputs"

# 정확한 버전
# python -m deep_learning.core.feature_shap_bmaml \
#   --ckpt "/Users/velocitygoal/Desktop/battery_project/v11/deep_learning/core_/core_checkpoints/cacle_bmaml_best_re_.pt" \
#   --background 50 \
#   --n_explain 40 \
#   --nsamples 80 \
#   --keep_last_t 50 \
#   --outdir "/Users/velocitygoal/Desktop/battery_project/v11/shap_outputs"
