#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
NASA 메타 DB에서
- battery / cycle / 40개 feature / rul_cycles 테이블을 CSV로 뽑고
- 각 feature vs rul_cycles 상관 / 회귀계수 / 통계 + SHAP importance를 계산해서
  CSV + JSON으로 저장하는 스크립트.

실행 예시 (core 디렉토리에서):

  cd /Users/velocitygoal/Desktop/battery_project/v11/deep_learning/core

  python export_nasa_feature_rul_stats.py \
    --ckpt ../core_checkpoints/nasa_bmaml_best_re.pt

결과 파일:
  - analysis/nasa_features_rul.csv
  - analysis/feature_rul_stats.csv
  - analysis/feature_rul_stats.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# -------------------------
# 경로 / sys.path 설정

# -------------------------
FILE_DIR = Path(__file__).resolve().parent          # .../deep_learning/core
PROJECT_ROOT = FILE_DIR.parent.parent               # .../v11

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# 내부 모듈
from deep_learning.core.meta_db_loader import load_cycle_db_for_meta
from deep_learning.core.prefix_inference_viz_meta_restored_v3_pyc import load_cfg_from_ckpt

ANALYSIS_DIR = FILE_DIR / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

SHAP_JSON_PATH = FILE_DIR / "shap_outputs" / "bmaml_shap_seq_feature_importance.json"


# -------------------------
# SHAP 로더 (선택)
# -------------------------
def load_shap_importance(path: Path):
    """
    bmaml_shap_seq_feature_importance.json 로드.
    없으면 빈 dict 반환.
    """
    if not path.exists():
        print(f"[WARN] SHAP json not found: {path}")
        return {}

    with open(path, "r") as f:
        data = json.load(f)

    names = data.get("feature_names", [])
    vals = data.get("importance", [])
    shap_map = {}
    for n, v in zip(names, vals):
        try:
            shap_map[str(n)] = float(v)
        except Exception:
            continue

    print(f"[INFO] Loaded SHAP importance for {len(shap_map)} features.")
    return shap_map


# -------------------------
# 메인 로직
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="/Users/velocitygoal/Desktop/battery_project/v11/deep_learning/core/core_checkpoints/nasa_bmaml_best_re.pt",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.ckpt).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"[LOAD] Checkpoint: {ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")

    # 1) 학습 당시 config 로드
    cfg = load_cfg_from_ckpt(ckpt["config"])

    # 2) NASA 메타 DB 로드 (train+val+test filter가 로더 안에서 적용됨)
    df, target_col, feature_cols, bid_col, cyc_col = load_cycle_db_for_meta(cfg)

    print(f"[INFO] Raw meta DB shape: {df.shape}")
    print(f"[INFO] Target col : {target_col}")
    print(f"[INFO] Feature cols ({len(feature_cols)}): {feature_cols}")
    print(f"[INFO] Battery col: {bid_col}, Cycle col: {cyc_col}")

    # 3) battery / cycle / features / RUL만 뽑기
    cols_export = [bid_col, cyc_col] + feature_cols + [target_col]
    df_export = df[cols_export].copy()

    # NaN 가진 행은 일단 그대로 두고, 나중에 통계 계산 시 drop
    out_csv_path = ANALYSIS_DIR / "nasa_features_rul.csv"
    df_export.to_csv(out_csv_path, index=False)
    print(f"[SAVE] Exported feature+RUL table to {out_csv_path}")

    # 4) 통계 계산용: NaN 있는 행은 제거
    df_stats = df_export.dropna(subset=[target_col] + feature_cols)
    y = df_stats[target_col].astype(float)
    X = df_stats[feature_cols].astype(float)

    print(f"[INFO] Stats df shape after NaN drop: {df_stats.shape}")
    print("[INFO] RUL basic stats:")
    print(y.describe())

    # 5) SHAP importance 로드 (선택)
    shap_map = load_shap_importance(SHAP_JSON_PATH)

    # 6) feature별 corr / slope / mean / std 계산
    stats_rows = []
    y_mean = y.mean()
    y_std = y.std(ddof=0)  # population std

    for feat in feature_cols:
        x = X[feat].astype(float)
        # NaN 방지
        mask = x.notna() & y.notna()
        if mask.sum() < 10:
            # 데이터 너무 적으면 건너뜀
            continue

        x_valid = x[mask]
        y_valid = y[mask]

        x_mean = x_valid.mean()
        x_std = x_valid.std(ddof=0)

        # 상관계수
        if x_std == 0 or y_std == 0:
            corr = np.nan
            slope = np.nan
        else:
            corr = float(np.corrcoef(x_valid, y_valid)[0, 1])
            # 단순 1D 선형 회귀 기울기: slope = cov(x,y) / var(x)
            cov = float(((x_valid - x_mean) * (y_valid - y_mean)).mean())
            var_x = float(((x_valid - x_mean) ** 2).mean())
            slope = cov / var_x if var_x > 0 else np.nan

        stats_rows.append(
            {
                "feature": feat,
                "mean": float(x_mean),
                "std": float(x_std),
                "corr_with_rul": float(corr) if not np.isnan(corr) else np.nan,
                "slope_rul_per_unit": float(slope) if not np.isnan(slope) else np.nan,
                "shap_importance": float(shap_map.get(feat, 0.0)),
            }
        )

    stats_df = pd.DataFrame(stats_rows)
    if stats_df.empty:
        print("[ERROR] No stats rows computed. Check data.")
        return

    # 7) 저장 및 콘솔 출력
    stats_csv_path = ANALYSIS_DIR / "feature_rul_stats.csv"
    stats_json_path = ANALYSIS_DIR / "feature_rul_stats.json"

    stats_df.to_csv(stats_csv_path, index=False)
    print(f"[SAVE] Feature–RUL stats CSV to {stats_csv_path}")

    # JSON: feature를 key로 한 dict
    stats_dict = {row["feature"]: row for row in stats_rows}
    with open(stats_json_path, "w") as f:
        json.dump(stats_dict, f, indent=2)
    print(f"[SAVE] Feature–RUL stats JSON to {stats_json_path}")

    # SHAP 기준 상위 10
    print("\n[TOP 10 by SHAP importance]")
    top_shap = (
        stats_df.sort_values("shap_importance", ascending=False)
        .head(10)[["feature", "shap_importance", "corr_with_rul", "slope_rul_per_unit"]]
    )
    print(top_shap.to_string(index=False))

    # 상관계수 기준 상위/하위 10
    print("\n[TOP 10 negative corr] (값 커질수록 RUL 줄어드는 축)")
    top_neg = (
        stats_df.sort_values("corr_with_rul")
        .head(10)[["feature", "corr_with_rul", "slope_rul_per_unit"]]
    )
    print(top_neg.to_string(index=False))

    print("\n[TOP 10 positive corr] (값 커질수록 RUL 늘어나는 축)")
    top_pos = (
        stats_df.sort_values("corr_with_rul", ascending=False)
        .head(10)[["feature", "corr_with_rul", "slope_rul_per_unit"]]
    )
    print(top_pos.to_string(index=False))


if __name__ == "__main__":
    main()

# cd /Users/velocitygoal/Desktop/battery_project/v11/deep_learning/core

# python export_nasa_feature_rul_stats.py \
#   --ckpt core_checkpoints/nasa_bmaml_best_re.pt