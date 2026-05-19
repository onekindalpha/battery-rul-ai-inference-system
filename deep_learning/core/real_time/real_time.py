#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
real_time.py

CSS Electronics EV data pack 중
decoded_ev6_data_full.csv 를 간단히 분석하는 스크립트.

기능
------
1) CSV 로드 & 기본 요약
   - 전체 주행 거리, 시간, 평균 속도
   - 평균 SoC / SoH / 온도
   - 간단한 EFC(Equivalent Full Cycles) & RUL 근사

2) kWh/100km 효율 계산 (CSS EV6 케이스 스터디 방식)
   - Power[W] = BatteryDCVoltage * BatteryCurrent
   - Speed[m/s] = Speed[km/h] / 3.6
   - kWh/100km = (Power / Speed_mps) * 0.027777778

3) 날짜(하루) 단위 summary 테이블 생성 후 CSV로 저장
   - 파일명 예: ev6_daily_summary.csv

4) EV6 물리 파생 피처 추가
   - power_kw, eff_c_rate, dod_from_soc, temp_rise_ev6

사용법
------
$ cd /Users/velocitygoal/Desktop/battery_project/v11
$ python -m deep_learning.core.real_time \
    --csv /Users/velocitygoal/Desktop/battery_project/v11/deep_learning/core/real_time/decoded_ev6_data_full.csv \
    --out /Users/velocitygoal/Desktop/battery_project/v11/deep_learning/core/real_time/ev6_daily_summary.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------------
# EV6 관련 상수 (필요 시 나중에 CLI로 뺄 수 있음)
# -----------------------------------
USABLE_KWH = 74.6        # EV6 usable capacity (kWh) - GT/롱레인지 기준 근사
DESIGN_EFC_EOL = 1000.0  # 1000 EFC에서 SoH 80% 도달한다고 가정
SOH_EOL = 80.0           # EOL SoH (%)


# ---------------------------
# 0. EFC & RUL
# ---------------------------

def add_efc_and_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    EV6 로그에 등가 풀사이클(EFC)와 간단한 RUL 근사를 추가.

    - EFC = 누적 방전 에너지 / 사용 가능 용량
      * 누적 방전 에너지: CED_CumulativeEnergyDischarged [Wh] (CSV 기준)
      * USABLE_KWH: 사용 가능한 팩 용량 (kWh)

    - RUL_from_efc = 설계 수명(1000EFC) - 현재까지 EFC
    - RUL_from_soh = SoH 기준 남은 비율 * 설계 수명
      (SoH 80%를 EOL로 보고, 100%→80% 구간을 DESIGN_EFC_EOL로 매핑)

    - 최종 RUL_cycles = 위 둘 중 더 보수적인 값(min), 음수는 0으로 클램핑
    """
    df = df.copy()

    if "CED_CumulativeEnergyDischarged" not in df.columns:
        df["EFC"] = np.nan
        df["RUL_cycles"] = np.nan
        return df

    # --- EFC 계산 ---
    energy_wh = df["CED_CumulativeEnergyDischarged"].astype(float).to_numpy()
    mask = np.isfinite(energy_wh)
    if mask.any():
        base = np.nanmin(energy_wh[mask])
    else:
        base = 0.0

    # Wh → kWh → cycles
    efc = (energy_wh - base) / (USABLE_KWH * 1000.0)
    efc[~mask] = np.nan
    df["EFC"] = efc

    # --- EFC 기반 RUL ---
    rul_from_efc = DESIGN_EFC_EOL - efc

    # --- SoH 기반 RUL cap ---
    soh = df.get("StateOfHealth", pd.Series(np.nan, index=df.index)).astype(float).to_numpy()
    mask_soh = np.isfinite(soh)
    rul_from_soh = np.full_like(efc, np.nan)
    if mask_soh.any():
        # SoH 100% → frac=1, SoH_EOL → frac=0
        frac = (soh[mask_soh] - SOH_EOL) / (100.0 - SOH_EOL)
        frac = np.clip(frac, 0.0, 1.5)
        rul_from_soh[mask_soh] = DESIGN_EFC_EOL * frac

    # --- 둘 중 더 보수적인 값 사용 ---
    rul_cycles = rul_from_efc.copy()
    both = np.isfinite(rul_from_soh)
    rul_cycles[both] = np.minimum(rul_cycles[both], rul_from_soh[both])
    rul_cycles = np.maximum(0.0, rul_cycles)

    df["RUL_cycles"] = rul_cycles
    return df


# ---------------------------
# 1. 데이터 로더
# ---------------------------

def load_ev6_csv(csv_path: Path) -> pd.DataFrame:
    """
    decoded_ev6_data_full.csv 전용 로더.
    주요 컬럼만 usecols로 읽고, 빠진 컬럼은 NaN으로 채운다.
    """
    print(f"[INFO] Loading EV6 CSV from: {csv_path}")

    header_df = pd.read_csv(csv_path, nrows=0)
    all_cols = list(header_df.columns)
    print(f"[INFO] CSV columns ({len(all_cols)}): {all_cols}")

    preferred_cols = [
        "TimeStamp",
        "StateOfHealth",
        "StateOfChargeBMS",
        "BatteryDCVoltage",
        "BatteryCurrent",
        "Speed",
        "DistanceTotal",                 # 총 주행거리 (있으면 이걸 우선 사용)
        "DistanceTrip",
        "CED_CumulativeEnergyDischarged",
        "OutdoorTemperature",
        "BatteryMaxTemperature",
    ]

    existing_for_usecols = [c for c in preferred_cols if c in all_cols]

    if not existing_for_usecols:
        print("[WARN] Preferred columns not found, reading full CSV.")
        df = pd.read_csv(csv_path)
    else:
        print(f"[INFO] Using columns for read_csv: {existing_for_usecols}")
        df = pd.read_csv(csv_path, usecols=existing_for_usecols)

    # 빠진 컬럼은 NaN으로
    for col in preferred_cols:
        if col not in df.columns:
            print(f"[WARN] Column '{col}' not found in CSV. Filling with NaN.")
            df[col] = np.nan

    df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], errors="coerce")
    df = df.sort_values("TimeStamp").reset_index(drop=True)
    return df


# ---------------------------
# 2. 효율(kWh/100km) 계산
# ---------------------------

def add_efficiency_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    CSS EV6 케이스 스터디에 나온 방식으로 kWh/100km 계산.

        Power_W   = BatteryDCVoltage * BatteryCurrent
        Speed_mps = Speed_kmh / 3.6
        kWh_100km = (Power_W / Speed_mps) * 0.027777778

    0 km/h(정지) 구간은 NaN 처리해서 divide-by-zero 피함.
    """
    df = df.copy()

    required_cols = {"BatteryDCVoltage", "BatteryCurrent", "Speed"}
    if not required_cols.issubset(df.columns):
        print(f"[WARN] 효율 계산에 필요한 컬럼이 없습니다: {required_cols - set(df.columns)}")
        df["kWh_100km"] = np.nan
        return df

    speed_mps = df["Speed"].astype(float) / 3.6
    power_w = df["BatteryDCVoltage"].astype(float) * df["BatteryCurrent"].astype(float)

    # 0 km/h (정지 구간)은 NaN 처리
    speed_safe = speed_mps.replace(0, np.nan)

    df["kWh_100km"] = (power_w / speed_safe) * 0.027777778
    return df


# ---------------------------
# 2-1. EV6 물리 파생 피처
# ---------------------------

def add_ev6_derived_physics(
    df: pd.DataFrame,
    pack_ah: float = 120.6,   # EV6 롱레인지 기준 Ah
) -> pd.DataFrame:
    """
    EV6 로그에서 간단한 '물리 계열' 파생 피처를 추가:

      - power_kw      : 순간 전력 (kW)
      - eff_c_rate    : 순간 C-rate (|I| / Ah)
      - dod_from_soc  : SoC 기반 DoD (0~1)
      - temp_rise_ev6 : 배터리 온도 - 외기 온도
    """
    df = df.copy()

    # 1) 전력 kW
    if {"BatteryDCVoltage", "BatteryCurrent"}.issubset(df.columns):
        power_w = df["BatteryDCVoltage"].astype(float) * df["BatteryCurrent"].astype(float)
        df["power_kw"] = power_w / 1000.0
    else:
        df["power_kw"] = np.nan

    # 2) C-rate
    if "BatteryCurrent" in df.columns and pack_ah > 0:
        df["eff_c_rate"] = df["BatteryCurrent"].abs().astype(float) / float(pack_ah)
    else:
        df["eff_c_rate"] = np.nan

    # 3) DoD (SoC 기반)
    if "StateOfChargeBMS" in df.columns:
        soc_frac = df["StateOfChargeBMS"].astype(float) / 100.0
        df["dod_from_soc"] = 1.0 - soc_frac
    else:
        df["dod_from_soc"] = np.nan

    # 4) 온도 상승 (배터리 max - 외기)
    if "BatteryMaxTemperature" in df.columns and "OutdoorTemperature" in df.columns:
        df["temp_rise_ev6"] = (
            df["BatteryMaxTemperature"].astype(float)
            - df["OutdoorTemperature"].astype(float)
        )
    else:
        df["temp_rise_ev6"] = np.nan

    return df


# ---------------------------
# 3. 전체 요약 출력
# ---------------------------

def print_global_summary(df: pd.DataFrame) -> None:
    """EV6 전체 로그에 대한 간단 요약."""
    print("\n=== EV6 Global summary ===")

    # 시간 범위
    if "TimeStamp" in df.columns and not df["TimeStamp"].empty:
        t0 = df["TimeStamp"].min()
        t1 = df["TimeStamp"].max()
        duration = t1 - t0
        print(f"- Time range        : {t0}  ~  {t1}  (≈ {duration})")

    # 거리
    if "DistanceTotal" in df.columns and df["DistanceTotal"].notna().any():
        dist_m = df["DistanceTotal"].max() - df["DistanceTotal"].min()
        dist_km = dist_m / 1000.0
        print(f"- Total distance    : {dist_km:.1f} km (from DistanceTotal)")
    elif "DistanceTrip" in df.columns and df["DistanceTrip"].notna().any():
        dist_m = df["DistanceTrip"].max() - df["DistanceTrip"].min()
        dist_km = dist_m / 1000.0
        print(f"- Total distance    : {dist_km:.1f} km (from DistanceTrip)")

    # 평균 속도
    if "Speed" in df.columns:
        print(f"- Mean speed        : {df['Speed'].mean():.1f} km/h")

    # SoC / SoH
    if "StateOfChargeBMS" in df.columns:
        print(f"- Mean SoC (BMS)    : {df['StateOfChargeBMS'].mean():.1f} %")
    if "StateOfHealth" in df.columns:
        print(f"- Mean SoH          : {df['StateOfHealth'].mean():.1f} %")

    # 온도
    if "BatteryMaxTemperature" in df.columns:
        print(f"- Mean batt max temp: {df['BatteryMaxTemperature'].mean():.1f} °C")
    if "OutdoorTemperature" in df.columns:
        print(f"- Mean outside temp : {df['OutdoorTemperature'].mean():.1f} °C")

    # 효율
    if "kWh_100km" in df.columns:
        mean_eff = df["kWh_100km"].dropna().mean()
        print(f"- Mean efficiency   : {mean_eff:.1f} kWh / 100 km")

    # EFC & RUL
    if "EFC" in df.columns:
        total_efc = df["EFC"].max() - df["EFC"].min()
        print(f"- Total EFC         : {total_efc:.2f} equivalent full cycles")

    if "RUL_cycles" in df.columns:
        current_rul = df["RUL_cycles"].iloc[-1]
        print(f"- Current RUL (EFC) : {current_rul:.1f} cycles (approx)")

    print("===========================\n")


# ---------------------------
# 4. 날짜(하루) 단위 summary
# ---------------------------

def build_daily_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    날짜(캘린더 day) 기준으로 간단 Summary 테이블 생성.

    컬럼 예:
      - date
      - distance_km
      - Speed_mean
      - StateOfChargeBMS_mean
      - StateOfHealth_mean
      - OutdoorTemperature_mean
      - BatteryMaxTemperature_mean
      - kWh_100km_mean
    """
    if "TimeStamp" not in df.columns:
        raise ValueError("TimeStamp 컬럼이 필요합니다.")

    df = df.copy()
    df["date"] = df["TimeStamp"].dt.date

    agg_kwargs = {}

    # 거리: DistanceTotal이 있으면 그걸 우선 사용, 없으면 DistanceTrip
    if "DistanceTotal" in df.columns:
        agg_kwargs["distance_km"] = (
            "DistanceTotal",
            lambda s: float(s.max() - s.min()) / 1000.0,
        )
    elif "DistanceTrip" in df.columns:
        agg_kwargs["distance_km"] = (
            "DistanceTrip",
            lambda s: float(s.max() - s.min()) / 1000.0,
        )

    base_cols = [
        "Speed",
        "StateOfChargeBMS",
        "StateOfHealth",
        "OutdoorTemperature",
        "BatteryMaxTemperature",
        "kWh_100km",
    ]

    for c in base_cols:
        if c in df.columns:
            agg_kwargs[f"{c}_mean"] = (c, "mean")

    daily = df.groupby("date").agg(**agg_kwargs).reset_index()

    # 열 순서 정리
    cols = ["date"]
    if "distance_km" in daily.columns:
        cols.append("distance_km")
    cols += [c for c in daily.columns if c not in cols]

    return daily[cols]


# ---------------------------
# 5. CLI 파서 & 메인
# ---------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EV6 decoded CSV에서 일/전체 요약, EFC, RUL, 파생 피처를 계산합니다."
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="decoded_ev6_data_full.csv 경로",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="일자별 summary CSV를 저장할 경로 (예: ev6_daily_summary.csv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"입력 CSV를 찾을 수 없습니다: {csv_path}")

    out_path = Path(args.out)

    # 1) 로드
    df = load_ev6_csv(csv_path)

    # 2) 효율
    print("[INFO] Computing kWh/100km efficiency signal ...")
    df = add_efficiency_column(df)

    # 3) EFC & RUL
    print("[INFO] Computing EFC & simple RUL ...")
    df = add_efc_and_rul(df)

    # 4) EV6 파생 피처
    print("[INFO] Adding EV6 derived physics features (power, C-rate, DoD, temp_rise)...")
    df = add_ev6_derived_physics(df, pack_ah=120.6)

    # 5) 전체 요약 출력
    print_global_summary(df)

    # 6) 일자별 summary 저장
    print("[INFO] Building daily summary table ...")
    daily = build_daily_summary(df)
    daily.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[INFO] Saved daily summary to: {out_path}")
    print(daily.head())

    # 7) 원본 + 파생피처 통합 CSV도 같이 저장 (대시보드/EDA용)
    full_out = csv_path.with_name("ev6_with_efc_rul.csv")
    df.to_csv(full_out, index=False, encoding="utf-8")
    print(f"[INFO] Saved full EV6 with EFC/RUL/physics to: {full_out}")


if __name__ == "__main__":
    main()


# cd /Users/velocitygoal/Desktop/battery_project/v11/deep_learning/core

# cd /Users/velocitygoal/Desktop/battery_project/v11/deep_learning/core

# python -m deep_learning.core.real_time --csv /Users/velocitygoal/Desktop/battery_project/v11/deep_learning/core/real_time/decoded_ev6_data_full.csv --out /Users/velocitygoal/Desktop/battery_project/v11/deep_learning/core/real_time/ev6_daily_summary.csv

