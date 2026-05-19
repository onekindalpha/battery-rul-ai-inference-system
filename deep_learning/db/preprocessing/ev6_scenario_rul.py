from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd


# EV6 설계상 EOL로 가정하는 EFC
DESIGN_EFC_EOL: float = 1000.0


def load_feature_stats(path: Path) -> Dict[str, Any]:
    """
    NASA 쪽에서 export한 feature_rul_stats.json 로드.

    Args:
        path: feature_rul_stats.json 경로

    Returns:
        dict 형태의 통계 정보
    """
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_ev6_scenario_features(ev6_df: pd.DataFrame) -> Dict[str, float]:
    """
    EV6 로그 (예: ev6_with_efc_rul.csv 일부 구간)에 대해
    NASA 시나리오 엔진이 이해할 수 있는 피처들을 계산.

    반환:
        {
          "soh": 0~1 proxy,
          "regen_strength": float,
          "voltage_min": float,
        }
    """
    if ev6_df is None:
        raise ValueError("ev6_df is None")
    df = ev6_df.copy()

    if df.empty:
        raise ValueError("EV6 데이터가 비어 있습니다.")

    # --- soh proxy (0~1 스케일, EFC 기반) ---
    if "EFC" not in df.columns:
        raise ValueError("EV6 데이터에 'EFC' 컬럼이 없습니다. real_time.py 전처리로 EFC를 먼저 붙여야 합니다.")

    efc_vals = df["EFC"].astype(float).to_numpy()
    efc_vals = efc_vals[np.isfinite(efc_vals)]
    if efc_vals.size == 0:
        raise ValueError("EFC에 유효한 값이 없습니다.")

    mean_efc = float(np.mean(efc_vals))
    soh_proxy = 1.0 - mean_efc / DESIGN_EFC_EOL
    soh_proxy = float(np.clip(soh_proxy, 0.0, 1.0))

    # --- regen_strength proxy (power_kw < 0 비율) ---
    if "power_kw" in df.columns:
        power_kw = df["power_kw"].astype(float).to_numpy()
    else:
        if "BatteryDCVoltage" not in df.columns or "BatteryCurrent" not in df.columns:
            raise ValueError(
                "regen_strength 계산을 위해 power_kw 또는 (BatteryDCVoltage, BatteryCurrent)가 필요합니다."
            )
        v = df["BatteryDCVoltage"].astype(float).to_numpy()
        i = df["BatteryCurrent"].astype(float).to_numpy()
        power_kw = v * i / 1000.0

    mask_valid = np.isfinite(power_kw)
    if mask_valid.any():
        regen_flag = power_kw[mask_valid] < 0.0
        regen_ratio = float(np.mean(regen_flag))  # 0~1
        regen_strength = regen_ratio * 5.0        # NASA 스케일 대충 맞춰서 확대
    else:
        regen_strength = 0.0

    # --- voltage_min ---
    if "MinCellVoltage" in df.columns:
        v_min = float(df["MinCellVoltage"].astype(float).min())
    elif "BatteryDCVoltage" in df.columns:
        v_min = float(df["BatteryDCVoltage"].astype(float).min())
    else:
        raise ValueError("voltage_min 계산을 위해 MinCellVoltage 또는 BatteryDCVoltage 컬럼이 필요합니다.")

    return {
        "soh": soh_proxy,
        "regen_strength": float(regen_strength),
        "voltage_min": float(v_min),
    }


def estimate_rul_from_scenario(
    ev6_feats: Dict[str, float],
    feature_stats: Dict[str, Any],
    baseline_feats: Dict[str, float] | None = None,
    baseline_rul: float = DESIGN_EFC_EOL,
) -> Dict[str, float]:
    """
    feature_rul_stats.json 에서 slope_rul_per_unit를 가져와
    baseline RUL에서 EV6 시나리오 피처만큼 보정한 RUL을 근사.

    Args:
        ev6_feats: compute_ev6_scenario_features의 출력
        feature_stats: feature_rul_stats.json 로드 결과
        baseline_feats: 기준 피처 값 (없으면 stats의 mean 사용)
        baseline_rul: 기준 RUL (cycles)

    Returns:
        {
          "rul_scenario": float,   # 근사 RUL (cycles)
          "delta_rul": float,      # baseline 대비 변화량
        }
    """
    if feature_stats is None:
        raise ValueError("feature_stats is None (feature_rul_stats.json 로드 필요).")

    # baseline 피처가 없으면 NASA 통계 mean 사용
    if baseline_feats is None:
        baseline_feats = {}
        for feat_key, stats in feature_stats.items():
            mean_v = float(stats.get("mean", 0.0))
            baseline_feats[feat_key] = mean_v

    delta_rul = 0.0

    for feat_key, ev6_val in ev6_feats.items():
        stats = feature_stats.get(feat_key, {})
        slope = float(stats.get("slope_rul_per_unit", 0.0))

        base_v = float(baseline_feats.get(feat_key, stats.get("mean", 0.0)))
        delta_v = float(ev6_val) - base_v
        contrib = slope * delta_v
        delta_rul += contrib

    scen_rul = float(max(0.0, baseline_rul + delta_rul))

    return {
        "rul_scenario": scen_rul,
        "delta_rul": float(delta_rul),
    }


if __name__ == "__main__":
    """
    1단계: 모듈 단독 테스트 용도.

    - ev6_with_efc_rul.csv에서 최근 30일 구간을 뽑아서
    - EV6 시나리오 피처 계산
    - feature_rul_stats.json 로드
    - 시나리오 기반 RUL 근사값 출력
    """
    from pathlib import Path

    # 네 프로젝트 구조에 맞게 수정해야 할 수도 있음.
    # 일단 지금 업로드된 파일 기준으로 /mnt/data 경로 사용
    EV6_CSV = Path("/Users/velocitygoal/Desktop/battery_project/v11/deep_learning/core/real_time/ev6_with_efc_rul.csv")
    FEATURE_STATS_PATH = Path("/Users/velocitygoal/Desktop/battery_project/v11/deep_learning/core/analysis/feature_rul_stats.csv")  # 실제 경로에 맞게 바꿔줘야 함

    print("EV6_CSV:", EV6_CSV)
    print("FEATURE_STATS_PATH:", FEATURE_STATS_PATH)

    if not EV6_CSV.exists():
        print("⚠ EV6_CSV 파일을 찾을 수 없습니다. 경로를 실제 프로젝트 구조에 맞게 수정하세요.")
    else:
        df_ev6 = pd.read_csv(EV6_CSV, parse_dates=["TimeStamp"])
        df_ev6 = df_ev6.sort_values("TimeStamp")

        # 최근 30일 구간 (TimeStamp 없으면 이 부분도 나중에 조정 필요)
        if "TimeStamp" in df_ev6.columns:
            cutoff = df_ev6["TimeStamp"].max() - pd.Timedelta(days=30)
            recent_df = df_ev6[df_ev6["TimeStamp"] >= cutoff].copy()
        else:
            recent_df = df_ev6.copy()

        print("recent_df shape:", recent_df.shape)

        try:
            feats = compute_ev6_scenario_features(recent_df)
            print("EV6 scenario features:", feats)

            feature_stats_dict = load_feature_stats(FEATURE_STATS_PATH)

            res = estimate_rul_from_scenario(
                ev6_feats=feats,
                feature_stats=feature_stats_dict,
                baseline_feats=None,
                baseline_rul=DESIGN_EFC_EOL,
            )

            print("scenario-based RUL (cycles) =", res["rul_scenario"])
            print("delta RUL vs baseline =", res["delta_rul"])
        except Exception as e:
            print("❌ 에러 발생:", repr(e))
