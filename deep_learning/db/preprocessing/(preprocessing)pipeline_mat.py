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

OUT_DB_BASIC  = DB_DIR / "battery_training_data_cleaned_final_basic.csv"
OUT_DB_CAUSAL = DB_DIR / "battery_training_data_cleaned_final_causal.csv"
OUT_CEEMDAN   = DB_DIR / "ceemdan_all_batteries_advanced_causal.csv"

# Pipeline hyperparameters (externalized to avoid magic numbers)
# Values are chosen to be consistent with NASA-style battery RUL literature
# - EOL_SOH_THRESHOLD ~ 0.8 (80% of initial capacity)
PIPELINE_CONFIG: Dict[str, float] = {
    "K_REF": 10,                 # 초기 참조 window 크기 (capacity/impedance baseline)
    "W_SMOOTH": 5,               # causal median smoothing window
    "W_SLOPE": 8,                # slope_xy에서 사용하는 window 길이
    "ANCHOR_WINDOW_INIT": 10,    # anchor 탐색 초기 window
    "ANCHOR_WINDOW_ROLL": 10,    # anchor rolling median window
    "ANCHOR_STABLE_LEN": 5,      # 안정 구간 길이
    "ANCHOR_RATIO": 0.85,        # 초기 용량 대비 anchor threshold 비율
    "EPS": 1e-6,                 # 수치 안정성용 eps
    "EOL_SOH_THRESHOLD": 0.8,    # EOL SoH (예: 80%)
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
    """Return the first non-magic key from a loaded .mat dict."""
    keys = [k for k in d.keys() if not k.startswith("__")]
    if not keys:
        raise ValueError("No valid keys in mat file.")
    return keys[0]


def _safe_attr(x, name: str, default=None):
    """
    Safely access `x.name` or `x[name]` for MATLAB struct / dict hybrids.
    """
    if hasattr(x, name):
        return getattr(x, name)
    if isinstance(x, dict) and name in x:
        return x[name]
    return default


def load_one_mat(file_path: Path) -> pd.DataFrame:
    """
    Load NASA-style .mat and return per-discharge-cycle summary DataFrame.

    Rich per-cycle features (raw domain):

      keys:
        - battery, cycle_num

      capacity:
        - capacity_ahr          (end-of-discharge capacity, Ah)
        - capacity_mean         (mean of capacity array in cycle)

      voltage:
        - voltage_measured_mean
        - voltage_min, voltage_max, voltage_std
        - v_dod_10, v_dod_50, v_dod_90  (DoD 10/50/90%에서의 전압)

      temperature / environment:
        - temperature_measured_max
        - temperature_mean, temperature_min, temperature_std
        - ambient_temp_c
        - temp_rise_cycle = Tmax - ambient

      current / IR:
        - re_ohm_interp, rct_ohm_interp
        - ir_mean, ir_max
        - current_mean, current_std, current_min, current_max

      dynamics / operation:
        - discharge_time_sec
        - eff_c_rate     (capacity_ahr / discharge_time_hr, 평균 전류 proxy)
        - dvdt_max_abs   (max |dV/dt|)
        - dTdt_max       (max dT/dt)
    """
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

        # raw signals (NASA common names)
        cap  = _safe_attr(d, "Capacity", None)
        V    = _safe_attr(d, "Voltage_measured", None)
        T    = _safe_attr(d, "Temperature_measured", None)
        Time = _safe_attr(d, "Time", None)

        IR     = _safe_attr(d, "Internal_Resistance", None)
        V_load = _safe_attr(d, "Voltage_load", None)
        I_meas = _safe_attr(d, "Current_measured", None)
        I_load = _safe_attr(d, "Current_load", None)

        # cast to arrays safely
        cap_arr  = np.atleast_1d(cap).astype(float)  if cap  is not None else np.array([np.nan])
        V_arr    = np.atleast_1d(V).astype(float)    if V    is not None else np.array([np.nan])
        T_arr    = np.atleast_1d(T).astype(float)    if T    is not None else np.array([np.nan])
        t_arr    = np.atleast_1d(Time).astype(float) if Time is not None else np.array([np.nan])

        ir_arr    = np.atleast_1d(IR).astype(float)     if IR     is not None else np.array([np.nan])
        vload_arr = np.atleast_1d(V_load).astype(float) if V_load is not None else np.array([np.nan])
        imeas_arr = np.atleast_1d(I_meas).astype(float) if I_meas is not None else np.array([np.nan])
        iload_arr = np.atleast_1d(I_load).astype(float) if I_load is not None else np.array([np.nan])

        # ------------------------------------------------------------------
        # 1) Capacity (end-of-discharge 중심)
        # ------------------------------------------------------------------
        if np.any(np.isfinite(cap_arr)):
            finite_idx = np.where(np.isfinite(cap_arr))[0]
            cap_end = float(cap_arr[finite_idx[-1]])  # 마지막 finite 값
            cap_mean = float(np.nanmean(cap_arr))
        else:
            cap_end = np.nan
            cap_mean = np.nan
        capacity_ahr = cap_end  # 메인 용량 정의

        # ------------------------------------------------------------------
        # 2) Voltage stats
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # 3) Temperature / ambient
        # ------------------------------------------------------------------
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

# ---------------------------------------------------------------------
        # 4) DoD-based voltage (V at 10/50/90% of capacity span)
        #    NOTE:
        #    - NASA .mat에서 Capacity가 cycle-level scalar인 경우가 많음.
        #    - DoD 기반 전압을 계산하려면 cap_arr, V_arr가
        #      같은 길이의 time-series 여야 함.
        #    - 그렇지 않으면 그냥 NaN으로 두고 스킵.
        # ---------------------------------------------------------------------
        v_dod_10 = np.nan
        v_dod_50 = np.nan
        v_dod_90 = np.nan

        # 둘 다 time-series이고, 길이가 같고, 충분히 길 때만 계산
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
        # else: 기본 NaN 유지 (Capacity가 scalar거나 길이 mismatch인 경우)
        # ------------------------------------------------------------------
        # 5) Current stats
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # 6) IR stats + Re proxy
        # ------------------------------------------------------------------
        if np.any(np.isfinite(ir_arr)):
            ir_mean = float(np.nanmean(ir_arr))
            ir_max = float(np.nanmax(ir_arr))
        else:
            ir_mean = np.nan
            ir_max = np.nan

        # Re (series resistance) proxy
        # 1) use Internal_Resistance if exists
        # 2) else Ohm's law proxy: |V_meas_mean - V_load_mean| / |I_mean|
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

        rct_ohm = re_ohm  # NASA에는 Rct 없으니 동일 proxy 유지

        # ambient / Tmax fallback (기존 로직 유지)
        if not np.isfinite(ambient_temp_c):
            ambient_temp_c = 0.0
        if not np.isfinite(temperature_measured_max):
            temperature_measured_max = 0.0

        temp_rise_cycle = float(temperature_measured_max - ambient_temp_c)

        # ------------------------------------------------------------------
        # 7) Discharge time & effective C-rate proxy
        # ------------------------------------------------------------------
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
            eff_c_rate = float(capacity_ahr / discharge_time_hr)  # 평균 전류 proxy
        else:
            eff_c_rate = np.nan

        # ------------------------------------------------------------------
        # 8) dV/dt, dT/dt dynamics
        # ------------------------------------------------------------------
        dvdt_max_abs = np.nan
        dTdt_max = np.nan
        if len(t_arr) >= 2:
            dt = np.diff(t_arr).astype(float)

            # voltage derivative
            dV = np.diff(V_arr).astype(float)
            mask_v = np.isfinite(dt) & np.isfinite(dV) & (np.abs(dt) > 1e-6)
            if np.any(mask_v):
                dVdt = dV[mask_v] / dt[mask_v]
                if np.any(np.isfinite(dVdt)):
                    dvdt_max_abs = float(np.nanmax(np.abs(dVdt)))

            # temperature derivative
            dT = np.diff(T_arr).astype(float)
            mask_t = np.isfinite(dt) & np.isfinite(dT) & (np.abs(dt) > 1e-6)
            if np.any(mask_t):
                dTdt = dT[mask_t] / dt[mask_t]
                if np.any(np.isfinite(dTdt)):
                    dTdt_max = float(np.nanmax(dTdt))

        rows.append({
            "battery": battery_id,
            "cycle_num": discharge_idx,

            # capacity
            "capacity_ahr": capacity_ahr,
            "capacity_mean": cap_mean,

            # voltage
            "voltage_measured_mean": voltage_measured_mean,
            "voltage_min": voltage_min,
            "voltage_max": voltage_max,
            "voltage_std": voltage_std,
            "v_dod_10": v_dod_10,
            "v_dod_50": v_dod_50,
            "v_dod_90": v_dod_90,

            # temperature / env
            "temperature_measured_max": temperature_measured_max,
            "temperature_mean": temperature_mean,
            "temperature_min": temperature_min,
            "temperature_std": temperature_std,
            "ambient_temp_c": ambient_temp_c,
            "temp_rise_cycle": temp_rise_cycle,

            # impedance / IR
            "re_ohm_interp": re_ohm,
            "rct_ohm_interp": rct_ohm,
            "ir_mean": ir_mean,
            "ir_max": ir_max,

            # current
            "current_mean": current_mean,
            "current_std": current_std,
            "current_min": current_min,
            "current_max": current_max,

            # dynamics / operation
            "discharge_time_sec": discharge_time_sec,
            "eff_c_rate": eff_c_rate,
            "dvdt_max_abs": dvdt_max_abs,
            "dTdt_max": dTdt_max,
        })

    return pd.DataFrame(rows)


def build_raw_db_from_mat(mat_dir: Path) -> pd.DataFrame:
    """
    Parse all .mat files under `mat_dir` into a single per-cycle DataFrame.
    """
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

    # numeric & basic cleaning
    for col in df.columns:
        if col in ("battery",):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=["capacity_ahr"]).reset_index(drop=True)
    df = df.fillna(0.0)
    df = df[df["capacity_ahr"] > 0].reset_index(drop=True)

    return df

# =============================================================================
# 2. Causal physics features + RUL computation
# =============================================================================

def add_physics_features_causal(
    df: pd.DataFrame,
    config: Dict[str, float] = PIPELINE_CONFIG,
) -> pd.DataFrame:
    """
    Add causal physics features on top of per-cycle basic features.

    - SoH, regen_strength, cap_vel, capacity_derivative
    - impedance_sum, impedance_growth, dcr, dcr_growth
    - LLI / LAM proxies (slope-based)
    - thermal_stress
    - RUL: cycle_life, rul_cycles, rul_norm (EOL threshold on SoH)
    """
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

    # 공통 winsorize
    def winsorize(x: np.ndarray, lo: float = 1, hi: float = 99) -> np.ndarray:
        a, b = np.percentile(x, lo), np.percentile(x, hi)
        return np.clip(x, a, b)

    # causal rolling helper (median)
    def causal_roll(x: np.ndarray, W_local: int) -> np.ndarray:
        y = np.zeros_like(x)
        for t in range(len(x)):
            s = max(0, t - W_local + 1)
            y[t] = np.median(x[s:t+1])
        return y

    # slope dY/dX over last W_slope points
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
        """
        Find an anchor cycle index based on capacity stability.
        Ensures we start analysis from a relatively stable capacity region.
        """
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
        """
        Backward-looking linear slope over last W_local points.
        """
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
        # 1) anchor 찾기
        C0 = g["capacity_ahr"].astype(float).values
        anchor = find_anchor_cycle(C0)

        # anchor 이전 cycle은 버리기
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

        # ---- causal smoothing (measurement noise 억제)
        C  = causal_roll(C_raw,  W)
        V  = causal_roll(V_raw,  W)
        Re = causal_roll(Re_raw, W)
        Rct = causal_roll(Rct_raw, W)

        k0 = min(k_ref, len(g))
        C_ref0  = np.nanmedian(C[:k0])
        V_ref0  = np.nanmedian(V[:k0])
        Re_ref0 = np.nanmedian(Re[:k0])

        # running best 용량
        C_ref_run = np.maximum.accumulate(np.nan_to_num(C, nan=C_ref0))

        # 임피던스
        Z = Re + Rct
        Z_ref0 = np.nanmedian(Z[:k0])

        # 델타
        dC  = C_ref_run - C
        dV  = V_ref0 - V
        dRe = Re - Re_ref0

        # dC floor (regen 구간 숫자 폭발 방지)
        pos_dC = dC[dC > 0]
        if len(pos_dC) > 0:
            dC_floor = np.percentile(pos_dC[:k0], 5)
            dC_floor = max(dC_floor, 1e-3)
        else:
            dC_floor = 1e-3
        _ = np.maximum(dC, dC_floor)  # denom (현재는 직접 사용 X)

        # SoH
        soh = C / (C_ref_run + eps)
        soh = np.clip(soh, 0.0, 1.0)

        # 용량 기울기
        cap_vel = past_slope(C, W)
        cap_deriv = np.diff(C, prepend=C[0])

        # 임피던스 growth (퍼센트 + winsorize + log)
        dcr_t = Re

        # log(Z / Z_ref0), log(Re / Re_ref0) 형태
        imp_growth = np.log((Z     + eps) / (Z_ref0  + eps))
        dcr_growth = np.log((dcr_t + eps) / (Re_ref0 + eps))

        # 너무 극단적인 값 한 번 더 자르기 (per-battery)
        imp_growth = winsorize(imp_growth, 1, 99)
        dcr_growth = winsorize(dcr_growth, 1, 99)

        # log-ratio 자체를 최종 피처로 사용
        imp_growth_log = imp_growth
        dcr_growth_log = dcr_growth

        # LLI / LAM: window slope 기반 (dY/dX)
        lli_slope = slope_xy(dC,  dV,  W_slope)
        lam_slope = slope_xy(dC,  dRe, W_slope)

        lli_w = winsorize(lli_slope, 1, 99)
        lam_w = winsorize(lam_slope, 1, 99)
        # --- lam 추가 안정화 (heavy tail 컷)
        lam_w = np.clip(lam_w, -100.0, 100.0)
        # thermal
        if "temperature_measured_max" in g.columns and "ambient_temp_c" in g.columns:
            Tmax = g["temperature_measured_max"].astype(float).values
            Tamb = g["ambient_temp_c"].astype(float).values
            dT = Tmax - Tamb
            Tref = np.nanmedian(Tmax[:k0])
            alpha = 0.05  # could be externalized if needed
            thermal_stress = np.exp(alpha * (Tmax - Tref))
        else:
            dT = np.zeros(len(g))
            thermal_stress = np.zeros(len(g))

        # ---- RUL: cycle_life, rul_cycles, rul_norm ----
        cycle_nums = g["cycle_num"].astype(int).values
        eol_mask = soh <= eol_soh_thr
        if np.any(eol_mask):
            eol_idx = int(np.where(eol_mask)[0][0])
            cycle_life = int(cycle_nums[eol_idx])
        else:
            # 데이터가 EOL까지 가지 않으면 NaN으로 두고, downstream에서 처리
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

        # write back
        g["soh"] = soh
        g["capacity_derivative"] = cap_deriv
        g["cap_vel"] = cap_vel
        g["regen_strength"] = dC  # running-best 대비 drop

        g["impedance_sum"] = Z
        g["impedance_growth"] = imp_growth
        g["impedance_growth_log"] = imp_growth_log

        g["dcr"] = dcr_t
        g["dcr_growth"] = dcr_growth
        g["dcr_growth_log"] = dcr_growth_log

        g["lli"] = lli_w
        g["lam"] = lam_w
        g["lam"] = g["lam"].clip(-100, 100)
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
# 3. CEEMDAN (same as before, minor cleanup)
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
            # Fallback to linear interpolation
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

        # pad IMFs to length 6 for consistent downstream usage
        while len(imfs) < 6:
            imfs.append(np.zeros_like(signal))

        return imfs, residual


def build_ceemdan(df: pd.DataFrame, max_imfs: int = 6) -> pd.DataFrame:
    """
    Build CEEMDAN-based features (IMF1-IMF6, Residual) from capacity_ahr time series.
    """
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
# 4. Run
# =============================================================================

def main() -> None:
    print("=" * 80)
    print("[Pipeline RE] MAT -> basic DB + causal DB + CEEMDAN (+ RUL)")
    print(f"MAT_DIR: {MAT_DIR}")
    print("=" * 80)

    # 1) MAT -> basic per-cycle summary (NASA 기반)
    raw_df = build_raw_db_from_mat(MAT_DIR)

    # basic DB: NASA-style per-cycle summary (rich) 그대로 저장
    df_basic = raw_df.copy()
    df_basic.to_csv(OUT_DB_BASIC, index=False)
    print(f"[DONE] basic DB saved -> {OUT_DB_BASIC}")

    # 2) basic 위에 causal physics + RUL features 추가
    df_causal = add_physics_features_causal(df_basic, config=PIPELINE_CONFIG)

    # global winsorize for some unstable features (추가 안정화)
    def global_winsorize(series: pd.Series, lo: float = 1, hi: float = 99) -> pd.Series:
        a, b = np.percentile(series.values, lo), np.percentile(series.values, hi)
        return series.clip(a, b)

    for col in ["dcr_growth", "impedance_growth", "lli", "lam"]:
        if col in df_causal.columns:
            df_causal[col] = global_winsorize(df_causal[col], 1, 99)

    # causal DB에 포함할 컬럼들만 선택
    KEEP_COLS = [
        # keys
        "battery", "cycle_num",
        "anchor_cycle",

        # raw / basic rich features
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

        # causal official physics columns
        "soh",
        "capacity_derivative",
        "cap_vel",
        "regen_strength",
        "impedance_sum",
        "impedance_growth",
        # "impedance_growth_log",
        "dcr",
        "dcr_growth",
        # "dcr_growth_log",
        "lli",
        "lam",
        "temp_rise",
        "thermal_stress",

        # RUL-related
        "cycle_life",
        "rul_cycles",
        "rul_norm",
    ]
    KEEP_COLS = [c for c in KEEP_COLS if c in df_causal.columns]
    df_causal = df_causal[KEEP_COLS].copy()

    df_causal.to_csv(OUT_DB_CAUSAL, index=False)
    print(f"[DONE] causal DB saved -> {OUT_DB_CAUSAL}")

    # 3) CEEMDAN: basic DB 기준 capacity_ahr 시계열로 IMFs 계산
    ceemdan_df = build_ceemdan(df_basic, max_imfs=6)
    ceemdan_df.to_csv(OUT_CEEMDAN, index=False)
    print(f"[DONE] ceemdan saved -> {OUT_CEEMDAN}")

    print("=" * 80)
    print("✅ Finished")
    print("=" * 80)


if __name__ == "__main__":
    main()
