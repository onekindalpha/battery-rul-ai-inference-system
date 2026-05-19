"""
pipeline_1_re.py (MAT-based, basic + causal physics DB rebuild)

- Input:
    deep_learning/db/mat/*.mat  (NASA battery .mat style)

- Output1 (basic):
    deep_learning/db/battery_training_data_cleaned_final_basic.csv
    -> NASA raw를 cycle-level로 요약한 기본 + 확장 피처 포함

- Output2 (causal):
    deep_learning/db/battery_training_data_cleaned_final_causal.csv
    -> basic 위에 인위적으로 만든 물리 피처(soh, regen_strength, lli, lam, 성장률 등) + RUL 추가

- Output3 (CEEMDAN):
    deep_learning/db/ceemdan_all_batteries_advanced_causal.csv
    -> capacity_ahr 시계열 기반 CEEMDAN (basic/casual과 같은 cycle 축)
"""

import os
import logging
from pathlib import Path
from typing import Dict
import math
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import argrelextrema
from scipy.interpolate import CubicSpline

# =============================================================================
# 0. Paths & Config
# =============================================================================
FILE_PATH = Path(__file__).resolve()
ROOT_DIR = FILE_PATH.parent
while not (ROOT_DIR / "db").exists() and ROOT_DIR != ROOT_DIR.parent:
    ROOT_DIR = ROOT_DIR.parent
DB_DIR = ROOT_DIR / "db"
MAT_DIR = DB_DIR / "mat"

# 기존 NASA 출력
OUT_DB_BASIC  = DB_DIR / "battery_training_data_cleaned_final_basic.csv"
OUT_DB_CAUSAL = DB_DIR / "battery_training_data_cleaned_final_causal.csv"
OUT_CEEMDAN   = DB_DIR / "ceemdan_all_batteries_advanced_causal.csv"

# ---- CACLE 전용 입력/출력 추가 ----
CACLE_ROOT   = ROOT_DIR / "test_cacle_dataset" / "cacle_dataset"
CACLE_SUBDIRS = ["CS2_35-2", "CS2_36-2", "CS2_37-2", "CS2_38-2"]

CACLE_OUT_DB_BASIC  = DB_DIR / "cacle_battery_training_data_cleaned_final_basic.csv"
CACLE_OUT_DB_CAUSAL = DB_DIR / "cacle_battery_training_data_cleaned_final_causal.csv"
CACLE_OUT_CEEMDAN   = DB_DIR / "cacle_ceemdan_all_batteries_advanced_causal.csv"

# Pipeline hyperparameters
PIPELINE_CONFIG: Dict[str, float] = {
    "K_REF": 10,
    "W_SMOOTH": 5,
    "W_SLOPE": 8,
    "ANCHOR_WINDOW_INIT": 10,
    "ANCHOR_WINDOW_ROLL": 10,
    "ANCHOR_STABLE_LEN": 5,
    "ANCHOR_RATIO": 0.85,
    "EPS": 1e-6,
    "EOL_SOH_THRESHOLD": 0.8,
}

# Basic logger
logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# =============================================================================
# 1. Robust MAT Loader (NASA style, rich cycle-level features)
# =============================================================================

def _get_first_data_key(d: dict) -> str:
    keys = [k for k in d.keys() if not k.startswith("__")]
    if not keys:
        raise ValueError("No valid keys in mat file.")
    return keys[0]


def _safe_attr(x, name: str, default=None):
    if hasattr(x, name):
        return getattr(x, name)
    if isinstance(x, dict) and name in x:
        return x[name]
    return default


def load_one_mat(file_path: Path) -> pd.DataFrame:
    m = loadmat(str(file_path), squeeze_me=True, struct_as_record=False)
    data_key = _get_first_data_key(m)
    batt = m[data_key]

    battery_id = file_path.stem
    cycles = _safe_attr(batt, "cycle", None)
    if cycles is None:
        raise ValueError(f"MAT structure does not have 'cycle': {file_path}")

    rows = []
    discharge_idx = 0

    for cyc in np.atleast_1d(cycles):
        ctype = str(_safe_attr(cyc, "type", "")).lower()
        if "discharge" not in ctype:
            continue

        discharge_idx += 1
        d = _safe_attr(cyc, "data", None)
        if d is None:
            logger.warning("Cycle with no data in %s, skipping.", file_path.name)
            continue

        cap  = _safe_attr(d, "Capacity", None)
        V    = _safe_attr(d, "Voltage_measured", None)
        T    = _safe_attr(d, "Temperature_measured", None)
        Time = _safe_attr(d, "Time", None)

        IR     = _safe_attr(d, "Internal_Resistance", None)
        V_load = _safe_attr(d, "Voltage_load", None)
        I_meas = _safe_attr(d, "Current_measured", None)
        I_load = _safe_attr(d, "Current_load", None)

        cap_arr  = np.atleast_1d(cap).astype(float)  if cap  is not None else np.array([np.nan])
        V_arr    = np.atleast_1d(V).astype(float)    if V    is not None else np.array([np.nan])
        T_arr    = np.atleast_1d(T).astype(float)    if T    is not None else np.array([np.nan])
        t_arr    = np.atleast_1d(Time).astype(float) if Time is not None else np.array([np.nan])

        ir_arr    = np.atleast_1d(IR).astype(float)     if IR     is not None else np.array([np.nan])
        vload_arr = np.atleast_1d(V_load).astype(float) if V_load is not None else np.array([np.nan])
        imeas_arr = np.atleast_1d(I_meas).astype(float) if I_meas is not None else np.array([np.nan])
        iload_arr = np.atleast_1d(I_load).astype(float) if I_load is not None else np.array([np.nan])

        # 1) Capacity
        if np.any(np.isfinite(cap_arr)):
            finite_idx = np.where(np.isfinite(cap_arr))[0]
            cap_end = float(cap_arr[finite_idx[-1]])
            cap_mean = float(np.nanmean(cap_arr))
        else:
            cap_end = np.nan
            cap_mean = np.nan
        capacity_ahr = cap_end

        # 2) Voltage stats
        if np.any(np.isfinite(V_arr)):
            voltage_measured_mean = float(np.nanmean(V_arr))
            voltage_min = float(np.nanmin(V_arr))
            voltage_max = float(np.nanmax(V_arr))
            voltage_std = float(np.nanstd(V_arr))
        else:
            voltage_measured_mean = np.nan
            voltage_min = np.nan
            voltage_max = np.nan
            voltage_std = np.nan

        # 3) Temperature / ambient
        if np.any(np.isfinite(T_arr)):
            temperature_measured_max = float(np.nanmax(T_arr))
            temperature_mean = float(np.nanmean(T_arr))
            temperature_min = float(np.nanmin(T_arr))
            temperature_std = float(np.nanstd(T_arr))

            if np.isfinite(T_arr[0]):
                ambient_temp_c = float(T_arr[0])
            else:
                ambient_temp_c = float(np.nanmedian(T_arr))
        else:
            temperature_measured_max = np.nan
            temperature_mean = np.nan
            temperature_min = np.nan
            temperature_std = np.nan
            ambient_temp_c = np.nan

        # 4) DoD-based voltage
        v_dod_10 = np.nan
        v_dod_50 = np.nan
        v_dod_90 = np.nan
        if (
            cap_arr is not None
            and V_arr is not None
            and cap_arr.shape == V_arr.shape
            and cap_arr.size >= 3
        ):
            mask = np.isfinite(cap_arr) & np.isfinite(V_arr)
            if np.sum(mask) >= 3:
                cap_f = cap_arr[mask]
                v_f = V_arr[mask]
                cap_start = float(np.nanmin(cap_f))
                cap_end_norm = float(np.nanmax(cap_f))
                denom = cap_end_norm - cap_start
                if denom > 1e-6:
                    rel = (cap_f - cap_start) / denom
                    order = np.argsort(rel)
                    rel_sorted = rel[order]
                    v_sorted = v_f[order]
                    targets = np.array([0.1, 0.5, 0.9])
                    v_at = np.interp(targets, rel_sorted, v_sorted)
                    v_dod_10, v_dod_50, v_dod_90 = [float(v) for v in v_at]

        # 5) Current stats
        current_mean = np.nan
        current_std = np.nan
        current_min = np.nan
        current_max = np.nan

        if np.any(np.isfinite(imeas_arr)):
            i_use = imeas_arr
        elif np.any(np.isfinite(iload_arr)):
            i_use = iload_arr
        else:
            i_use = None

        if i_use is not None and np.any(np.isfinite(i_use)):
            current_mean = float(np.nanmean(i_use))
            current_std = float(np.nanstd(i_use))
            current_min = float(np.nanmin(i_use))
            current_max = float(np.nanmax(i_use))

        # 6) IR stats + Re proxy
        if np.any(np.isfinite(ir_arr)):
            ir_mean = float(np.nanmean(ir_arr))
            ir_max = float(np.nanmax(ir_arr))
        else:
            ir_mean = np.nan
            ir_max = np.nan

        if np.any(np.isfinite(ir_arr)):
            re_ohm = float(np.nanmean(ir_arr))
        else:
            v_load_mean = float(np.nanmean(vload_arr)) if np.any(np.isfinite(vload_arr)) else np.nan

            if np.any(np.isfinite(imeas_arr)):
                i_mean_for_re = float(np.nanmean(imeas_arr))
            elif np.any(np.isfinite(iload_arr)):
                i_mean_for_re = float(np.nanmean(iload_arr))
            else:
                i_mean_for_re = np.nan

            if (
                np.isfinite(voltage_measured_mean)
                and np.isfinite(v_load_mean)
                and np.isfinite(i_mean_for_re)
                and abs(i_mean_for_re) > 1e-6
            ):
                re_ohm = abs(voltage_measured_mean - v_load_mean) / (abs(i_mean_for_re) + 1e-6)
            else:
                re_ohm = np.nan

        if not np.isfinite(re_ohm):
            re_ohm = 1e-3

        rct_ohm = re_ohm

        if not np.isfinite(ambient_temp_c):
            ambient_temp_c = 0.0
        if not np.isfinite(temperature_measured_max):
            temperature_measured_max = 0.0

        temp_rise_cycle = float(temperature_measured_max - ambient_temp_c)

        # 7) Discharge time & effective C-rate
        if np.any(np.isfinite(t_arr)):
            discharge_time_sec = float(np.nanmax(t_arr) - np.nanmin(t_arr))
        else:
            discharge_time_sec = np.nan

        if (
            np.isfinite(discharge_time_sec)
            and discharge_time_sec > 1e-6
            and np.isfinite(capacity_ahr)
            and capacity_ahr > 0
        ):
            discharge_time_hr = discharge_time_sec / 3600.0
            eff_c_rate = float(capacity_ahr / discharge_time_hr)
        else:
            eff_c_rate = np.nan

        # 8) dV/dt, dT/dt
        dvdt_max_abs = np.nan
        dTdt_max = np.nan
        if len(t_arr) >= 2:
            dt = np.diff(t_arr).astype(float)

            dV = np.diff(V_arr).astype(float)
            mask_v = np.isfinite(dt) & np.isfinite(dV) & (np.abs(dt) > 1e-6)
            if np.any(mask_v):
                dVdt = dV[mask_v] / dt[mask_v]
                if np.any(np.isfinite(dVdt)):
                    dvdt_max_abs = float(np.nanmax(np.abs(dVdt)))

            dT = np.diff(T_arr).astype(float)
            mask_t = np.isfinite(dt) & np.isfinite(dT) & (np.abs(dt) > 1e-6)
            if np.any(mask_t):
                dTdt = dT[mask_t] / dt[mask_t]
                if np.any(np.isfinite(dTdt)):
                    dTdt_max = float(np.nanmax(dTdt))

        rows.append({
            "battery": battery_id,
            "cycle_num": discharge_idx,
            "capacity_ahr": capacity_ahr,
            "capacity_mean": cap_mean,
            "voltage_measured_mean": voltage_measured_mean,
            "voltage_min": voltage_min,
            "voltage_max": voltage_max,
            "voltage_std": voltage_std,
            "v_dod_10": v_dod_10,
            "v_dod_50": v_dod_50,
            "v_dod_90": v_dod_90,
            "temperature_measured_max": temperature_measured_max,
            "temperature_mean": temperature_mean,
            "temperature_min": temperature_min,
            "temperature_std": temperature_std,
            "ambient_temp_c": ambient_temp_c,
            "temp_rise_cycle": temp_rise_cycle,
            "re_ohm_interp": re_ohm,
            "rct_ohm_interp": rct_ohm,
            "ir_mean": ir_mean,
            "ir_max": ir_max,
            "current_mean": current_mean,
            "current_std": current_std,
            "current_min": current_min,
            "current_max": current_max,
            "discharge_time_sec": discharge_time_sec,
            "eff_c_rate": eff_c_rate,
            "dvdt_max_abs": dvdt_max_abs,
            "dTdt_max": dTdt_max,
        })

    return pd.DataFrame(rows)


def build_raw_db_from_mat(mat_dir: Path) -> pd.DataFrame:
    dfs = []
    for fp in sorted(mat_dir.glob("*.mat")):
        try:
            df_b = load_one_mat(fp)
            if len(df_b) > 2:
                dfs.append(df_b)
                logger.info("[OK] %s: %d discharge cycles", fp.name, len(df_b))
            else:
                logger.warning("[SKIP] %s: too few discharge cycles (%d)", fp.name, len(df_b))
        except Exception as e:
            logger.exception("[SKIP] %s: %s", fp.name, e)
            continue

    if not dfs:
        raise RuntimeError("No valid MAT files parsed. Check MAT_DIR and file format.")

    df = pd.concat(dfs, ignore_index=True)

    for col in df.columns:
        if col in ("battery",):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=["capacity_ahr"]).reset_index(drop=True)
    df = df.fillna(0.0)  # NASA path: 기존 로직 유지
    df = df[df["capacity_ahr"] > 0].reset_index(drop=True)
    return df

# =============================================================================
# 1-bis. CACLE xlsx loader (cycle-level 형태라고 가정)
# =============================================================================

def load_one_cacle_xlsx(file_path: Path) -> pd.DataFrame:
    """
    CACLE Arbin xlsx 하나를 읽어서 per-cycle summary 로 변환.
    - Discharge_Capacity(Ah)는 파일 내에서 '누적' -> 각 사이클별 '증분(cap_cycle)'으로 변환
    - 동일 셀(CS2_35)은 여러 날짜 파일로 나뉘어 있어도, 여기서는 base id (예: CS2_35)까지만 붙임
    - 실제 global cycle_num은 build_raw_db_from_cacle_xlsx에서 재계산
    """
    xls = pd.ExcelFile(file_path)

    channel_sheets = [s for s in xls.sheet_names if s.lower().startswith("channel")]
    if not channel_sheets:
        sheet_name = xls.sheet_names[0] if len(xls.sheet_names) == 1 else xls.sheet_names[1]
    else:
        sheet_name = channel_sheets[0]

    df = pd.read_excel(file_path, sheet_name=sheet_name)

    rows = []

    # 예: CS2_35_10_15_10 -> battery_id = CS2_35
    stem = file_path.stem
    parts = stem.split("_")
    if len(parts) >= 2:
        battery_id = "_".join(parts[:2])
    else:
        battery_id = stem

    # 이 파일 안에서의 누적 capacity 기준 offset
    prev_cap_end = 0.0

    for cyc_idx, g in df.groupby("Cycle_Index"):
        g = g.copy()

        if "Discharge_Capacity(Ah)" not in g.columns:
            continue

        gd = g[(g["Current(A)"] < 0) | (g["Discharge_Capacity(Ah)"] > 0)]
        if gd.empty:
            continue

        if "Test_Time(s)" in gd.columns:
            gd = gd.sort_values("Test_Time(s)")
            t_arr = gd["Test_Time(s)"].to_numpy(dtype=float)
        else:
            t_arr = np.arange(len(gd), dtype=float)

        v_arr   = gd["Voltage(V)"].to_numpy(dtype=float)
        i_arr   = gd["Current(A)"].to_numpy(dtype=float)
        cap_arr = gd["Discharge_Capacity(Ah)"].to_numpy(dtype=float)
        ir_arr  = gd.get("Internal_Resistance(Ohm)", pd.Series(np.nan, index=gd.index)).to_numpy(dtype=float)

        # ---- (A) 누적 -> 사이클 단위 capacity로 변환 ----
        finite_cap = np.where(np.isfinite(cap_arr))[0]
        if finite_cap.size > 0:
            cap_end_cum = float(cap_arr[finite_cap[-1]])   # 이 사이클 끝 시점의 누적값
            capacity_ahr = cap_end_cum - prev_cap_end      # 한 사이클 동안 방전한 용량
            prev_cap_end = cap_end_cum

            # 파일 리셋/이상 등으로 0 이하 나오면 이 사이클은 스킵
            if not np.isfinite(capacity_ahr) or capacity_ahr <= 0:
                continue

            # mean은 그냥 해당 사이클 capacity 값으로 둠 (누적 mean은 의미가 애매함)
            capacity_mean = capacity_ahr
        else:
            continue

        # ---- Voltage stats ----
        finite_v = np.isfinite(v_arr)
        if finite_v.any():
            v_mean = float(np.nanmean(v_arr))
            v_min  = float(np.nanmin(v_arr))
            v_max  = float(np.nanmax(v_arr))
            v_std  = float(np.nanstd(v_arr))
        else:
            v_mean = v_min = v_max = v_std = math.nan

        # ---- Temperature: CACLE에는 실측 없음 → NaN 유지 ----
        temperature_measured_max = math.nan
        temperature_mean = math.nan
        temperature_min = math.nan
        temperature_std = math.nan
        ambient_temp_c = math.nan
        temp_rise_cycle = math.nan

        # ---- Current stats ----
        finite_i = np.isfinite(i_arr)
        if finite_i.any():
            current_mean = float(np.nanmean(i_arr))
            current_std  = float(np.nanstd(i_arr))
            current_min  = float(np.nanmin(i_arr))
            current_max  = float(np.nanmax(i_arr))
        else:
            current_mean = current_std = current_min = current_max = math.nan

        # ---- IR stats + Re proxy ----
        finite_ir = np.isfinite(ir_arr)
        if finite_ir.any():
            ir_mean = float(np.nanmean(ir_arr))
            ir_max  = float(np.nanmax(ir_arr))
            re_ohm  = float(np.nanmean(ir_arr))
        else:
            ir_mean = ir_max = math.nan
            re_ohm  = 1e-3  # 최소값 fallback

        rct_ohm = re_ohm

        # ---- Discharge time & C-rate ----
        if len(t_arr) >= 1 and np.isfinite(t_arr).any():
            discharge_time_sec = float(np.nanmax(t_arr) - np.nanmin(t_arr))
        else:
            discharge_time_sec = math.nan

        if (
            discharge_time_sec > 1e-6
            and math.isfinite(capacity_ahr)
            and capacity_ahr > 0
        ):
            eff_c_rate = float(capacity_ahr / (discharge_time_sec / 3600.0))
        else:
            eff_c_rate = math.nan

        # ---- dV/dt (dT/dt는 온도 없음 → NaN) ----
        if len(t_arr) >= 2:
            dt = np.diff(t_arr)
            dV = np.diff(v_arr)
            mask = np.isfinite(dt) & np.isfinite(dV) & (np.abs(dt) > 1e-6)
            if mask.any():
                dVdt = dV[mask] / dt[mask]
                dvdt_max_abs = float(np.nanmax(np.abs(dVdt)))
            else:
                dvdt_max_abs = math.nan
        else:
            dvdt_max_abs = math.nan
        dTdt_max = math.nan

        rows.append({
            "battery": battery_id,           # ★ base id (CS2_35)
            "source_file": stem,             # ★ 어느 xlsx에서 온 건지 보존
            "cycle_idx_local": int(cyc_idx), # ★ 파일 내부 cycle index (나중에 global 재정렬용)
            "capacity_ahr": capacity_ahr,
            "capacity_mean": capacity_mean,
            "voltage_measured_mean": v_mean,
            "voltage_min": v_min,
            "voltage_max": v_max,
            "voltage_std": v_std,
            "v_dod_10": math.nan,
            "v_dod_50": math.nan,
            "v_dod_90": math.nan,
            "temperature_measured_max": temperature_measured_max,
            "temperature_mean": temperature_mean,
            "temperature_min": temperature_min,
            "temperature_std": temperature_std,
            "ambient_temp_c": ambient_temp_c,
            "temp_rise_cycle": temp_rise_cycle,
            "re_ohm_interp": re_ohm,
            "rct_ohm_interp": rct_ohm,
            "ir_mean": ir_mean,
            "ir_max": ir_max,
            "current_mean": current_mean,
            "current_std": current_std,
            "current_min": current_min,
            "current_max": current_max,
            "discharge_time_sec": discharge_time_sec,
            "eff_c_rate": eff_c_rate,
            "dvdt_max_abs": dvdt_max_abs,
            "dTdt_max": dTdt_max,
        })

    return pd.DataFrame(rows)

def _parse_cacle_date_ord(stem: str) -> int:
    """
    파일명에서 날짜 순서를 추출해서 정렬용 정수로 변환.
    예: CS2_35_10_15_10 -> month=10, day=15, year=2010 -> 20101015
    """
    parts = stem.split("_")
    if len(parts) >= 5:
        try:
            month = int(parts[2])
            day   = int(parts[3])
            year2 = int(parts[4])
            year  = 2000 + year2
            return year * 10000 + month * 100 + day
        except Exception:
            return 0
    return 0


def build_raw_db_from_cacle_xlsx(
    cacle_root: Path,
    subdirs: list[str],
) -> pd.DataFrame:
    dfs = []

    for sub in subdirs:
        subdir_path = cacle_root / sub
        if not subdir_path.exists():
            logger.warning("[CACLE] subdir does not exist: %s", subdir_path)
            continue

        for fp in sorted(subdir_path.glob("*.xlsx")):
            try:
                df_b = load_one_cacle_xlsx(fp)
                if len(df_b) > 0:
                    dfs.append(df_b)
                    logger.info("[CACLE OK] %s: %d discharge cycles", fp.name, len(df_b))
                else:
                    logger.warning("[CACLE SKIP] %s: no valid discharge cycles", fp.name)
            except Exception as e:
                logger.exception("[CACLE SKIP] %s: %s", fp.name, e)
                continue

    if not dfs:
        raise RuntimeError("No valid CACLE xlsx files parsed. Check CACLE_ROOT/CACLE_SUBDIRS and file format.")

    df = pd.concat(dfs, ignore_index=True)

    # 숫자형 변환 & 기본 클리닝
    for col in df.columns:
        if col in ("battery", "source_file"):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=["capacity_ahr"]).reset_index(drop=True)
    df = df[df["capacity_ahr"] > 0].reset_index(drop=True)

    # ---- (B) 동일 셀 단위로 전체 cycle 시퀀스 재정렬 ----
    # 파일명에서 날짜 순서를 뽑아서, (battery, date, local_cycle) 기준으로 sort
    df["date_ord"] = df["source_file"].apply(_parse_cacle_date_ord)
    df = df.sort_values(["battery", "date_ord", "cycle_idx_local"]).reset_index(drop=True)

    # 각 배터리별로 global cycle_num 1,2,3,... 부여
    df["cycle_num"] = df.groupby("battery").cumcount() + 1

    # 중간 작업용 컬럼 제거 (최종 DB에는 필요없음)
    df = df.drop(columns=["date_ord", "cycle_idx_local", "source_file"])

    return df

# =============================================================================
# 2. Causal physics features + RUL computation
# =============================================================================

def add_physics_features_causal(
    df: pd.DataFrame,
    config: Dict[str, float] = PIPELINE_CONFIG,
) -> pd.DataFrame:
    df = df.sort_values(["battery", "cycle_num"]).copy()
    out = []

    k_ref = int(config["K_REF"])
    W = int(config["W_SMOOTH"])
    W_slope = int(config["W_SLOPE"])
    anchor_win_init = int(config["ANCHOR_WINDOW_INIT"])
    anchor_win_roll = int(config["ANCHOR_WINDOW_ROLL"])
    anchor_stable_len = int(config["ANCHOR_STABLE_LEN"])
    anchor_ratio = float(config["ANCHOR_RATIO"])
    eps = float(config["EPS"])
    eol_soh_thr = float(config["EOL_SOH_THRESHOLD"])

    def winsorize(x: np.ndarray, lo: float = 1, hi: float = 99) -> np.ndarray:
        a, b = np.percentile(x, lo), np.percentile(x, hi)
        return np.clip(x, a, b)

    def causal_roll(x: np.ndarray, W_local: int) -> np.ndarray:
        y = np.zeros_like(x)
        for t in range(len(x)):
            s = max(0, t - W_local + 1)
            y[t] = np.median(x[s:t+1])
        return y

    def slope_xy(x: np.ndarray, y: np.ndarray, W_local: int) -> np.ndarray:
        s_arr = np.zeros_like(x)
        for t in range(len(x)):
            s = max(0, t - W_local + 1)
            xx = x[s:t+1]
            yy = y[s:t+1]
            if len(xx) < 2 or np.allclose(xx, xx[0]):
                s_arr[t] = 0.0
            else:
                s_arr[t] = np.polyfit(xx, yy, 1)[0]
        return s_arr

    def find_anchor_cycle(cap: np.ndarray) -> int:
        n = len(cap)
        if n == 0:
            return 1
        w0 = min(anchor_win_init, n)
        base_med = np.nanmedian(cap[:w0])
        if not np.isfinite(base_med) or base_med <= 0:
            return 1
        thr = anchor_ratio * base_med

        roll = np.zeros(n)
        for t in range(n):
            s = max(0, t - anchor_win_roll + 1)
            roll[t] = np.nanmedian(cap[s:t+1])

        for t in range(n - anchor_stable_len + 1):
            if np.all(roll[t:t+anchor_stable_len] >= thr):
                return t + 1
        return 1

    def past_slope(x: np.ndarray, W_local: int) -> np.ndarray:
        s_arr = np.zeros_like(x)
        for t in range(len(x)):
            s = max(0, t - W_local + 1)
            yy = x[s:t+1]
            if len(yy) < 2:
                s_arr[t] = 0.0
            else:
                tt = np.arange(len(yy))
                s_arr[t] = np.polyfit(tt, yy, 1)[0]
        return s_arr

    for bid, g in df.groupby("battery"):
        C0 = g["capacity_ahr"].astype(float).values
        anchor = find_anchor_cycle(C0)

        mask = g["cycle_num"].values >= anchor
        if not mask.any():
            logger.warning("[CAUSAL] battery=%s has no usable cycles after anchor=%d", bid, anchor)
            continue

        g = g.loc[mask].reset_index(drop=True)
        g["anchor_cycle"] = anchor

        C_raw  = g["capacity_ahr"].astype(float).values
        V_raw  = g["voltage_measured_mean"].astype(float).values
        Re_raw = g["re_ohm_interp"].astype(float).values
        if "rct_ohm_interp" in g.columns:
            Rct_raw = g["rct_ohm_interp"].astype(float).values
        else:
            Rct_raw = np.zeros_like(Re_raw)

        C  = causal_roll(C_raw,  W)
        V  = causal_roll(V_raw,  W)
        Re = causal_roll(Re_raw, W)
        Rct = causal_roll(Rct_raw, W)

        k0 = min(k_ref, len(g))
        C_ref0  = np.nanmedian(C[:k0])
        V_ref0  = np.nanmedian(V[:k0])
        Re_ref0 = np.nanmedian(Re[:k0])

        C_ref_run = np.maximum.accumulate(np.nan_to_num(C, nan=C_ref0))

        Z = Re + Rct
        Z_ref0 = np.nanmedian(Z[:k0])

        dC  = C_ref_run - C
        dV  = V_ref0 - V
        dRe = Re - Re_ref0

        pos_dC = dC[dC > 0]
        if len(pos_dC) > 0:
            dC_floor = np.percentile(pos_dC[:k0], 5)
            dC_floor = max(dC_floor, 1e-3)
        else:
            dC_floor = 1e-3
        _ = np.maximum(dC, dC_floor)

        soh = C / (C_ref_run + eps)
        soh = np.clip(soh, 0.0, 1.0)

        cap_vel = past_slope(C, W)
        cap_deriv = np.diff(C, prepend=C[0])

        dcr_t = Re

        imp_growth = np.log((Z     + eps) / (Z_ref0  + eps))
        dcr_growth = np.log((dcr_t + eps) / (Re_ref0 + eps))

        imp_growth = winsorize(imp_growth, 1, 99)
        dcr_growth = winsorize(dcr_growth, 1, 99)

        imp_growth_log = imp_growth
        dcr_growth_log = dcr_growth

        lli_slope = slope_xy(dC,  dV,  W_slope)
        lam_slope = slope_xy(dC,  dRe, W_slope)

        lli_w = winsorize(lli_slope, 1, 99)
        lam_w = winsorize(lam_slope, 1, 99)
        lam_w = np.clip(lam_w, -100.0, 100.0)

        # ---- thermal: 온도 데이터가 실제로 있을 때만 계산, 없으면 NaN 유지 ----
        if "temperature_measured_max" in g.columns and "ambient_temp_c" in g.columns:
            Tmax = g["temperature_measured_max"].astype(float).values
            Tamb = g["ambient_temp_c"].astype(float).values
            finite_mask = np.isfinite(Tmax) & np.isfinite(Tamb)

            if np.any(finite_mask):
                dT = Tmax - Tamb
                # 앞쪽 k0 구간에서 finite 값으로 기준 온도 계산
                Tmax_ref_candidates = Tmax[finite_mask][:k0]
                if np.any(np.isfinite(Tmax_ref_candidates)):
                    Tref = np.nanmedian(Tmax_ref_candidates)
                else:
                    Tref = np.nanmedian(Tmax[finite_mask])
                alpha = 0.05
                thermal_stress = np.exp(alpha * (Tmax - Tref))
            else:
                dT = np.full(len(g), np.nan)
                thermal_stress = np.full(len(g), np.nan)
        else:
            dT = np.full(len(g), np.nan)
            thermal_stress = np.full(len(g), np.nan)

        # ---- RUL: cycle_life, rul_cycles, rul_norm ----
        cycle_nums = g["cycle_num"].astype(int).values
        eol_mask = soh <= eol_soh_thr
        if np.any(eol_mask):
            eol_idx = int(np.where(eol_mask)[0][0])
            cycle_life = int(cycle_nums[eol_idx])
        else:
            cycle_life = np.nan
            logger.warning("[RUL] battery=%s never reaches SoH <= %.2f (len=%d)", bid, eol_soh_thr, len(soh))

        if np.isfinite(cycle_life):
            rul_cycles = np.maximum(cycle_life - cycle_nums, 0).astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                if cycle_life > 0:
                    rul_norm = rul_cycles / float(cycle_life)
                else:
                    rul_norm = np.full_like(rul_cycles, np.nan, dtype=float)
        else:
            rul_cycles = np.full_like(cycle_nums, np.nan, dtype=float)
            rul_norm = np.full_like(cycle_nums, np.nan, dtype=float)

        logger.info(
            "[CAUSAL] battery=%s: anchor=%d, n_cycles=%d, cycle_life=%s",
            bid, anchor, len(g), str(cycle_life),
        )

        g["soh"] = soh
        g["capacity_derivative"] = cap_deriv
        g["cap_vel"] = cap_vel
        g["regen_strength"] = dC
        g["impedance_sum"] = Z
        g["impedance_growth"] = imp_growth
        g["impedance_growth_log"] = imp_growth_log
        g["dcr"] = dcr_t
        g["dcr_growth"] = dcr_growth
        g["dcr_growth_log"] = dcr_growth_log
        g["lli"] = lli_w
        g["lam"] = lam_w
        g["temp_rise"] = dT
        g["thermal_stress"] = thermal_stress
        g["cycle_life"] = cycle_life
        g["rul_cycles"] = rul_cycles
        g["rul_norm"] = rul_norm

        out.append(g)

    if not out:
        raise RuntimeError("add_physics_features_causal produced empty output.")

    return pd.concat(out, ignore_index=True)

# =============================================================================
# 3. CEEMDAN
# =============================================================================

class AdvancedCEEMDAN:
    def __init__(self, noise_mode: str = "adaptive", n_ensembles: int = 150, seed: int = 42):
        self.noise_mode = noise_mode
        self.n_ensembles = n_ensembles
        np.random.seed(seed)

    def analyze_signal_characteristics(self, signal: np.ndarray) -> Dict[str, np.ndarray]:
        signal_mean = np.mean(signal)
        signal_std = np.std(signal)
        signal_normalized = (signal - signal_mean) / (signal_std + 1e-10)
        signal_diff = np.diff(signal_normalized)
        volatility = np.std(signal_diff)
        skewness = np.abs(np.mean(signal_diff ** 3) / (volatility ** 3 + 1e-10))
        return {
            "mean": signal_mean,
            "std": signal_std,
            "normalized": signal_normalized,
            "volatility": volatility,
            "skewness": skewness,
        }

    def set_adaptive_noise(self, signal_chars: Dict[str, np.ndarray]) -> float:
        volatility = signal_chars["volatility"]
        skewness = signal_chars["skewness"]
        base_noise = 0.2 * volatility
        base_noise *= (1 + 0.3 * np.tanh(skewness))
        return float(np.clip(base_noise, 0.01, 0.3))

    def find_extrema_advanced(self, signal: np.ndarray, order: int = 2):
        if len(signal) < 5:
            return np.array([], dtype=int), np.array([], dtype=int)

        maxima_idx = argrelextrema(signal, np.greater, order=order)[0]
        minima_idx = argrelextrema(signal, np.less, order=order)[0]
        return maxima_idx, minima_idx

    def get_smooth_envelope(self, signal: np.ndarray, extrema_idx: np.ndarray, kind: str = "max") -> np.ndarray:
        if len(extrema_idx) < 2:
            fill_val = np.max(signal) if kind == "max" else np.min(signal)
            return np.full_like(signal, fill_val)
        try:
            cs = CubicSpline(extrema_idx, signal[extrema_idx], bc_type="natural", extrapolate="linear")
            return cs(np.arange(len(signal)))
        except Exception:
            return np.interp(np.arange(len(signal)), extrema_idx, signal[extrema_idx])

    def sift_advanced(self, signal: np.ndarray, max_iterations: int = 200) -> np.ndarray:
        residual = signal.copy()
        for _ in range(max_iterations):
            maxima_idx, minima_idx = self.find_extrema_advanced(residual, order=2)
            if len(maxima_idx) < 2 or len(minima_idx) < 2:
                break
            upper_env = self.get_smooth_envelope(residual, maxima_idx, "max")
            lower_env = self.get_smooth_envelope(residual, minima_idx, "min")
            mean_env = (upper_env + lower_env) / 2.0
            residual_new = residual - mean_env
            sd = np.sum((residual_new - residual) ** 2) / (np.sum(residual ** 2) + 1e-10)
            if sd < 1e-4:
                break
            residual = residual_new
        return residual

    def decompose(self, signal: np.ndarray, max_imfs: int = 6):
        signal_chars = self.analyze_signal_characteristics(signal)
        signal_normalized = signal_chars["normalized"]
        signal_std = signal_chars["std"]

        if self.noise_mode == "adaptive":
            noise_std = self.set_adaptive_noise(signal_chars)
        else:
            noise_std = 0.2

        n_samples = len(signal_normalized)
        imfs_raw = []
        residual = signal_normalized.copy()

        for _ in range(max_imfs):
            if np.std(residual) < 0.01 * np.std(signal_normalized):
                break
            imf_ensemble = np.zeros((self.n_ensembles, n_samples))
            for ens in range(self.n_ensembles):
                noise = np.random.normal(0, noise_std, n_samples)
                signal_with_noise = residual + noise
                imf_ensemble[ens] = self.sift_advanced(signal_with_noise, max_iterations=200)
            imf = np.mean(imf_ensemble, axis=0)
            imfs_raw.append(imf)
            residual = residual - imf

        imfs = [imf * signal_std for imf in imfs_raw]
        residual = residual * signal_std + signal_chars["mean"]

        while len(imfs) < 6:
            imfs.append(np.zeros_like(signal))

        return imfs, residual


def build_ceemdan(df: pd.DataFrame, max_imfs: int = 6) -> pd.DataFrame:
    ceemdan = AdvancedCEEMDAN(noise_mode="adaptive", n_ensembles=150, seed=42)
    results = []

    for battery_id, g in df.groupby("battery"):
        g = g.sort_values("cycle_num")
        signal = g["capacity_ahr"].values.astype(float)
        cycles = g["cycle_num"].values.astype(int)

        if len(signal) < 5:
            logger.warning("[CEEMDAN] battery=%s has too few points (%d), skipping.", battery_id, len(signal))
            continue

        imfs, residual = ceemdan.decompose(signal, max_imfs=max_imfs)

        for i, cyc in enumerate(cycles):
            row = {
                "battery": battery_id,
                "cycle_num": int(cyc),
                "Capacity": float(signal[i]),
                "Residual": float(residual[i]),
            }
            for j in range(6):
                row[f"IMF{j+1}"] = float(imfs[j][i])
            results.append(row)

    if not results:
        raise RuntimeError("build_ceemdan produced empty output.")

    return pd.DataFrame(results)

# =============================================================================
# 4. Run (xlsx only)
# =============================================================================

def main() -> None:
    print("=" * 80)
    print("[Pipeline RE] CACLE XLSX ONLY -> cacle_basic + cacle_causal + cacle_CEEMDAN")
    print(f"CACLE_ROOT: {CACLE_ROOT}")
    print("=" * 80)

    # 1) CACLE XLSX -> basic
    cacle_raw_df = build_raw_db_from_cacle_xlsx(CACLE_ROOT, CACLE_SUBDIRS)
    cacle_basic = cacle_raw_df.copy()
    cacle_basic.to_csv(CACLE_OUT_DB_BASIC, index=False)
    print(f"[DONE] CACLE basic DB saved -> {CACLE_OUT_DB_BASIC}")

    # 2) CACLE causal physics + RUL
    cacle_causal = add_physics_features_causal(cacle_basic, config=PIPELINE_CONFIG)

    def global_winsorize(series: pd.Series, lo: float = 1, hi: float = 99) -> pd.Series:
        a, b = np.percentile(series.values, lo), np.percentile(series.values, hi)
        return series.clip(a, b)

    for col in ["dcr_growth", "impedance_growth", "lli", "lam"]:
        if col in cacle_causal.columns:
            cacle_causal[col] = global_winsorize(cacle_causal[col], 1, 99)

    KEEP_COLS = [
        "battery", "cycle_num",
        "anchor_cycle",
        "capacity_ahr",
        "capacity_mean",
        "ambient_temp_c",
        "voltage_measured_mean",
        "voltage_min",
        "voltage_max",
        "voltage_std",
        "v_dod_10",
        "v_dod_50",
        "v_dod_90",
        "re_ohm_interp",
        "rct_ohm_interp",
        "ir_mean",
        "ir_max",
        "temperature_measured_max",
        "temperature_mean",
        "temperature_min",
        "temperature_std",
        "temp_rise_cycle",
        "discharge_time_sec",
        "eff_c_rate",
        "current_mean",
        "current_std",
        "current_min",
        "current_max",
        "dvdt_max_abs",
        "dTdt_max",
        "soh",
        "capacity_derivative",
        "cap_vel",
        "regen_strength",
        "impedance_sum",
        "impedance_growth",
        "dcr",
        "dcr_growth",
        "lli",
        "lam",
        "temp_rise",
        "thermal_stress",
        "cycle_life",
        "rul_cycles",
        "rul_norm",
    ]
    KEEP_COLS_CACLE = [c for c in KEEP_COLS if c in cacle_causal.columns]
    cacle_causal = cacle_causal[KEEP_COLS_CACLE].copy()
    cacle_causal.to_csv(CACLE_OUT_DB_CAUSAL, index=False)
    print(f"[DONE] CACLE causal DB saved -> {CACLE_OUT_DB_CAUSAL}")

    # 3) CACLE CEEMDAN
    cacle_ceemdan = build_ceemdan(cacle_basic, max_imfs=6)
    cacle_ceemdan.to_csv(CACLE_OUT_CEEMDAN, index=False)
    print(f"[DONE] CACLE ceemdan saved -> {CACLE_OUT_CEEMDAN}")

    print("=" * 80)
    print("✅ Finished (CACLE ONLY)")
    print("=" * 80)


if __name__ == "__main__":
    main()
