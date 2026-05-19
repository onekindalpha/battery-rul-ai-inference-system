from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import json
import numpy as np
import pandas as pd

DESIGN_EFC_EOL: float = 1000.0

ALLOWED_FEATURES = {
    "temp_rise_cycle",
    "soh",
    "current_mean",
    "current_min",
    "regen_strength",
    "ambient_temp_c",
}

SIGN_OVERRIDES = {
    "soh": "pos",
    "ambient_temp_c": "neg",
    "temp_rise_cycle": "neg",
}


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


def _apply_sign_override(feat_key: str, slope_raw: float) -> float:
    """
    SIGN_OVERRIDES에 따라 slope 부호를 물리 직관에 맞게 강제.
    """
    mode = SIGN_OVERRIDES.get(feat_key)
    if mode is None:
        return slope_raw
    if mode == "neg":
        return -abs(slope_raw)
    if mode == "pos":
        return abs(slope_raw)
    return slope_raw


def compute_ev6_scenario_features(ev6_df: pd.DataFrame) -> Dict[str, float]:
    """
    EV6 로그 (예: ev6_with_efc_rul.csv 전체 또는 일부 구간)에 대해
    NASA 시나리오 엔진이 이해할 수 있는 피처들을 계산.

    반환 예:
        {
          "soh": 0~1,
          "regen_strength": ~0~2,
          "ambient_temp_c": °C,
          "temp_rise_cycle": °C,
          "current_mean": A,
          "current_min": A,
          # "thermal_stress": float (optional),
        }

    ev6_df는 보통 선택된 기간(df_full_sel) 같은 슬라이스를 넣으면 됨.
    """
    if ev6_df is None:
        raise ValueError("ev6_df is None")

    df = ev6_df.copy()
    if df.empty:
        raise ValueError("EV6 데이터가 비어 있습니다.")

    if "EFC" not in df.columns:
        raise ValueError(
            "EV6 데이터에 'EFC' 컬럼이 없습니다. real_time.py 전처리로 EFC를 먼저 붙여야 합니다."
        )

    efc_vals = df["EFC"].astype(float).to_numpy()
    efc_vals = efc_vals[np.isfinite(efc_vals)]
    if efc_vals.size == 0:
        raise ValueError("EFC에 유효한 값이 없습니다.")

    mean_efc = float(np.mean(efc_vals))
    soh_proxy = 1.0 - mean_efc / DESIGN_EFC_EOL
    soh_proxy = float(np.clip(soh_proxy, 0.0, 1.0))

    # regen_strength
    if "power_kw" in df.columns:
        power_kw = df["power_kw"].astype(float).to_numpy()
    else:
        if ("BatteryDCVoltage" not in df.columns) or ("BatteryCurrent" not in df.columns):
            raise ValueError("regen_strength 계산을 위해 power_kw 또는 (BatteryDCVoltage, BatteryCurrent)가 필요합니다.")
        v = df["BatteryDCVoltage"].astype(float).to_numpy()
        i = df["BatteryCurrent"].astype(float).to_numpy()
        power_kw = (v * i) / 1000.0
    mask_valid_p = np.isfinite(power_kw)
    if mask_valid_p.any():
        regen_flag = power_kw[mask_valid_p] < 0.0
        regen_ratio = float(np.mean(regen_flag))
        regen_strength = regen_ratio * 2.0
    else:
        regen_strength = 0.0

    # ambient_temp_c
    if "OutdoorTemperature" in df.columns:
        ambient_temp_c = float(df["OutdoorTemperature"].astype(float).mean())
    elif "BatteryMaxTemperature" in df.columns:
        ambient_temp_c = float(df["BatteryMaxTemperature"].astype(float).mean())
    else:
        ambient_temp_c = 25.0

    # temp_rise_cycle
    if "BatteryMaxTemperature" in df.columns:
        t_pack = df["BatteryMaxTemperature"].astype(float)
        if "OutdoorTemperature" in df.columns:
            t_amb = df["OutdoorTemperature"].astype(float)
        else:
            t_amb = pd.Series(ambient_temp_c, index=df.index)

        temp_rise = (t_pack - t_amb).to_numpy()
        temp_rise_valid = temp_rise[np.isfinite(temp_rise)]
        if temp_rise_valid.size > 0:
            temp_rise_cycle = float(np.mean(temp_rise_valid))
        else:
            temp_rise_cycle = 0.0
    else:
        temp_rise_cycle = 0.0

    # current stats
    if "BatteryCurrent" in df.columns:
        cur = df["BatteryCurrent"].astype(float).to_numpy()
        cur_valid = cur[np.isfinite(cur)]
        if cur_valid.size > 0:
            current_mean = float(np.mean(cur_valid))
            current_min = float(np.min(cur_valid))
        else:
            current_mean = 0.0
            current_min = 0.0
    else:
        current_mean = 0.0
        current_min = 0.0

    # optional thermal stress (computed but not returned in feats)
    thermal_stress_mean = 1.0
    if "BatteryMaxTemperature" in df.columns:
        T = df["BatteryMaxTemperature"].astype(float).to_numpy()
        T_valid = T[np.isfinite(T)]
        if T_valid.size > 0:
            T_ref = 25.0
            alpha = 0.05
            thermal_stress = np.exp(alpha * (T_valid - T_ref))
            thermal_stress_mean = float(np.mean(thermal_stress))

    feats = {
        "soh": soh_proxy,
        "regen_strength": float(regen_strength),
        "ambient_temp_c": ambient_temp_c,
        "temp_rise_cycle": temp_rise_cycle,
        "current_mean": current_mean,
        "current_min": current_min,
    }
    return feats


def estimate_rul_from_scenario(
    ev6_feats: Dict[str, float],
    feature_stats: Dict[str, Any],
    baseline_feats: Dict[str, float] | None = None,
    baseline_rul: float = DESIGN_EFC_EOL,
    efc_current: float | None = None,
    factor_min: float = 0.5,
    factor_max: float = 1.5,
) -> Dict[str, float]:
    """
    feature_rul_stats.json 에서 slope_rul_per_unit를 가져와
    baseline RUL에서 EV6 시나리오 피처만큼 보정한 RUL을 근사.

    + 추가:
      - scenario_rul_raw / baseline_rul 로 "배수(factor)"를 만들고
      - factor를 [factor_min, factor_max] 안에서 클램핑
      - efc_current(현재 EFC)를 함께 받아서 EFC 기반 RUL과 섞은 최종 RUL 계산

    Args:
        ev6_feats: compute_ev6_scenario_features의 결과 딕셔너리
        feature_stats: feature_rul_stats.json 로드 결과
        baseline_feats:
            - None이면 NASA 통계 mean을 baseline feature로 사용
            - dict를 넘기면 그 값을 baseline으로 사용
        baseline_rul:
            - 기준 RUL (cycles), 기본값은 DESIGN_EFC_EOL (=1000)
        efc_current:
            - 현재까지 쌓인 EFC (예: df["EFC"].max())
            - None이면 EFC 기반 RUL 대신 baseline_rul를 그대로 사용
        factor_min, factor_max:
            - scenario_rul_raw / baseline_rul 배수를 이 구간 안에 클램핑

    Returns:
        {
          "rul_scenario": float,        # 최종 blended RUL (cycles)
          "rul_scenario_raw": float,    # baseline + delta_rul (클램핑 전)
          "delta_rul": float,
          "factor_raw": float,          # raw 배수
          "factor_clipped": float,      # 클램핑된 배수
          "rul_from_efc": float,        # EFC만으로 계산한 RUL (있다면)
        }
    """
    if feature_stats is None:
        raise ValueError("feature_stats is None (feature_rul_stats.json 로드 필요).")

    if baseline_feats is None:
        baseline_feats = {}
        for feat_key, stats in feature_stats.items():
            mean_v = float(stats.get("mean", 0.0))
            baseline_feats[feat_key] = mean_v

    delta_rul = 0.0
    for feat_key, ev6_val in ev6_feats.items():
        if feat_key not in ALLOWED_FEATURES:
            continue
        stats = feature_stats.get(feat_key)
        if not stats:
            continue

        slope_raw = float(stats.get("slope_rul_per_unit", 0.0))
        if slope_raw == 0.0:
            continue

        slope = _apply_sign_override(feat_key, slope_raw)
        base_v = float(baseline_feats.get(feat_key, stats.get("mean", 0.0)))
        delta_v = float(ev6_val) - base_v
        contrib = slope * delta_v
        delta_rul += contrib

    scen_rul_raw = float(max(0.0, baseline_rul + delta_rul))

    if baseline_rul > 0.0:
        factor_raw = scen_rul_raw / float(baseline_rul)
    else:
        factor_raw = 1.0

    if factor_min <= 0.0:
        factor_min = 0.1
    if factor_max < factor_min:
        factor_max = factor_min

    factor_clipped = float(min(max(factor_raw, factor_min), factor_max))

    if efc_current is not None:
        rul_from_efc = float(max(0.0, baseline_rul - float(efc_current)))
    else:
        rul_from_efc = float(baseline_rul)

    rul_final = rul_from_efc * factor_clipped

    return {
        "rul_scenario": float(rul_final),
        "rul_scenario_raw": float(scen_rul_raw),
        "delta_rul": float(delta_rul),
        "factor_raw": float(factor_raw),
        "factor_clipped": float(factor_clipped),
        "rul_from_efc": float(rul_from_efc),
    }


def compute_rul_scenarios_from_efc(
    efc_current: float,
    eol_cycles: float = DESIGN_EFC_EOL,
    factors: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """
    EFC 기준 RUL을 베이스로 해서,
    운행/충전 조건에 따른 3단계 RUL 시나리오를 만들어줌.

    - worst  : 앞으로 조건이 빡세질 때 (고온 + 높은 C-rate + DCFC 잦음 등)
    - normal : 지금과 비슷한 조건을 유지할 때
    - best   : 더 조심해서 관리할 때 (SOC 관리 + 고온/저온 회피 등)

    기본 factor:
        worst  = 0.7  → 가혹하면 EFC 기반 RUL의 70% 정도만 간다고 가정
        normal = 1.0  → 지금 가정(EOL=1000−EFC) 그대로
        best   = 1.2  → 잘 관리하면 EFC 기반 RUL의 120%까지 노려볼 수 있다고 가정
    """
    if factors is None:
        factors = {
            "worst": 0.7,
            "normal": 1.0,
            "best": 1.2,
        }

    rul_base = max(0.0, float(eol_cycles) - float(efc_current))
    scenarios: Dict[str, float] = {}
    for name, f in factors.items():
        scenarios[name] = rul_base * float(f)
    return scenarios


if __name__ == "__main__":
    EV6_CSV = Path(
        "/Users/velocitygoal/Desktop/battery_project/v11/deep_learning/core/real_time/ev6_with_efc_rul.csv"
    )
    FEATURE_STATS_PATH = Path(
        "/Users/velocitygoal/Desktop/battery_project/v11/deep_learning/core/analysis/feature_rul_stats.json"
    )

    print("EV6_CSV:", EV6_CSV)
    print("FEATURE_STATS_PATH:", FEATURE_STATS_PATH)

    df_ev6 = pd.read_csv(EV6_CSV, parse_dates=["TimeStamp"]).sort_values("TimeStamp")
    print("EV6 df shape:", df_ev6.shape)

    ev6_feats = compute_ev6_scenario_features(df_ev6)
    print("EV6 scenario features:", ev6_feats)

    if "EFC" in df_ev6.columns:
        efc_current = float(df_ev6["EFC"].max())
    else:
        efc_current = None

    feature_stats_dict = load_feature_stats(FEATURE_STATS_PATH)

    res = estimate_rul_from_scenario(
        ev6_feats=ev6_feats,
        feature_stats=feature_stats_dict,
        baseline_feats=None,
        baseline_rul=DESIGN_EFC_EOL,
        efc_current=efc_current,
        factor_min=0.5,
        factor_max=1.0,
    )

    print("----- RUL summary -----")
    print("EFC current:", efc_current)
    print("RUL from EFC only (cycles):", res["rul_from_efc"])
    print("Scenario-based RUL (raw, cycles):", res["rul_scenario_raw"])
    print("Scenario factor (raw):", res["factor_raw"])
    print("Scenario factor (clipped):", res["factor_clipped"])
    print("Final blended RUL (cycles):", res["rul_scenario"])

    scenarios_3 = compute_rul_scenarios_from_efc(efc_current=efc_current, eol_cycles=DESIGN_EFC_EOL)

    print("----- 3-stage RUL scenarios (cycles) -----")
    print("Worst-case (harsh usage):", scenarios_3["worst"])
    print("Normal (current-like usage):", scenarios_3["normal"])
    print("Best-case (careful usage):", scenarios_3["best"])
