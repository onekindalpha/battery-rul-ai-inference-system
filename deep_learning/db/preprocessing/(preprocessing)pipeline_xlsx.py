"""
pipeline_1_re.py (MAT-based, causal physics DB rebuild)

- Input: deep_learning/db/mat/*.mat  (NASA battery .mat style)
- Output1: deep_learning/db/battery_training_data_cleaned_final_causal.csv
- Output2: deep_learning/db/ceemdan_all_batteries_advanced.csv  (kept)
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.interpolate import interp1d
from scipy.signal import argrelextrema
from scipy.interpolate import CubicSpline

# =============================================================================
# 0. Paths
# =============================================================================
FILE_PATH = Path(__file__).resolve()
ROOT_DIR = FILE_PATH.parent
while not (ROOT_DIR / "db").exists() and ROOT_DIR != ROOT_DIR.parent:
    ROOT_DIR = ROOT_DIR.parent
DB_DIR = ROOT_DIR / "db"
MAT_DIR = DB_DIR / "mat"

OUT_DB = DB_DIR / "battery_training_data_cleaned_final_causal.csv"
OUT_CEEMDAN = DB_DIR / "ceemdan_all_batteries_advanced_causal.csv"

# =============================================================================
# 1. Robust MAT Loader (NASA style)
# =============================================================================

def _get_first_data_key(d: dict):
    keys = [k for k in d.keys() if not k.startswith("__")]
    if not keys:
        raise ValueError("No valid keys in mat file.")
    return keys[0]

def _safe_attr(x, name, default=None):
    if hasattr(x, name):
        return getattr(x, name)
    if isinstance(x, dict) and name in x:
        return x[name]
    return default

def load_one_mat(file_path: Path) -> pd.DataFrame:
    """
    Load NASA-style .mat and return per-discharge-cycle summary df
    Columns produced (raw only):
      battery, cycle_num, capacity_ahr, voltage_measured_mean,
      temperature_measured_max, ambient_temp_c,
      re_ohm_interp, rct_ohm_interp, discharge_time_sec
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

        # per-cycle scalar summaries
        cap_val = float(np.nanmean(cap_arr))
        voltage_measured_mean = float(np.nanmean(V_arr))
        temperature_measured_max = float(np.nanmax(T_arr))
        ambient_temp_c = float(T_arr[0]) if np.isfinite(T_arr[0]) else float(np.nanmedian(T_arr))

        # -------------------------
        # Re (series resistance) proxy
        # 1) use Internal_Resistance if exists
        # 2) else Ohm's law proxy: |V_meas_mean - V_load_mean| / |I_mean|
        # -------------------------
        if np.any(np.isfinite(ir_arr)):
            re_ohm = float(np.nanmean(ir_arr))
        else:
            v_load_mean = float(np.nanmean(vload_arr)) if np.any(np.isfinite(vload_arr)) else np.nan

            if np.any(np.isfinite(imeas_arr)):
                i_mean = float(np.nanmean(imeas_arr))
            elif np.any(np.isfinite(iload_arr)):
                i_mean = float(np.nanmean(iload_arr))
            else:
                i_mean = np.nan

            if np.isfinite(v_load_mean) and np.isfinite(i_mean) and abs(i_mean) > 1e-6:
                re_ohm = abs(voltage_measured_mean - v_load_mean) / (abs(i_mean) + 1e-6)
            else:
                re_ohm = np.nan

        if not np.isfinite(re_ohm):
            re_ohm = 1e-3
        if not np.isfinite(ambient_temp_c):
            ambient_temp_c = 0.0
        if not np.isfinite(temperature_measured_max):
            temperature_measured_max = 0.0

        rct_ohm = re_ohm  # NASA에는 Rct 없으니 동일 proxy 유지

        # discharge time (seconds)
        if np.any(np.isfinite(t_arr)):
            discharge_time_sec = float(np.nanmax(t_arr) - np.nanmin(t_arr))
        else:
            discharge_time_sec = np.nan

        rows.append({
            "battery": battery_id,
            "cycle_num": discharge_idx,
            "capacity_ahr": cap_val,
            "voltage_measured_mean": voltage_measured_mean,
            "temperature_measured_max": temperature_measured_max,
            "ambient_temp_c": ambient_temp_c,
            "re_ohm_interp": re_ohm,
            "rct_ohm_interp": rct_ohm,
            "discharge_time_sec": discharge_time_sec,
        })

    return pd.DataFrame(rows)


def build_raw_db_from_mat(mat_dir: Path) -> pd.DataFrame:
    dfs = []
    for fp in sorted(mat_dir.glob("*.mat")):
        try:
            df_b = load_one_mat(fp)
            if len(df_b) > 2:
                dfs.append(df_b)
                print(f"[OK] {fp.name}: {len(df_b)} discharge cycles")
        except Exception as e:
            print(f"[SKIP] {fp.name}: {e}")
            continue

    if not dfs:
        raise RuntimeError("No valid MAT files parsed.")

    df = pd.concat(dfs, ignore_index=True)

    # numeric & basic cleaning
    for col in df.columns:
        if col in ("battery",):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=["capacity_ahr"]).reset_index(drop=True)
    df = df.fillna(0.0)

    return df

# =============================================================================
# 2. Causal physics features (official formulas we agreed)
# =============================================================================
def add_physics_features_causal(df, k_ref=10, W=5, eps=1e-6):
    df = df.sort_values(["battery", "cycle_num"]).copy()
    out = []

    for bid, g in df.groupby("battery"):
        g = g.sort_values("cycle_num").reset_index(drop=True)

        C  = g["capacity_ahr"].astype(float).values
        V  = g["voltage_measured_mean"].astype(float).values
        Re = g["re_ohm_interp"].astype(float).values
        Rct = g["rct_ohm_interp"].astype(float).values if "rct_ohm_interp" in g.columns else np.zeros_like(Re)

        # ---- robust baselines from early cycles
        k0 = min(k_ref, len(g))
        C_ref0  = np.nanmedian(C[:k0])
        V_ref0  = np.nanmedian(V[:k0])
        Re_ref0 = np.nanmedian(Re[:k0])

        # causal running capacity reference
        C_ref_run = np.maximum.accumulate(np.nan_to_num(C, nan=C_ref0))

        # impedance baseline
        Z = Re + Rct
        Z_ref0 = np.nanmedian(Z[:k0])

        # causal deltas
        dC  = C_ref_run - C
        dV  = V_ref0 - V
        dRe = Re - Re_ref0

        # ---- SoH(t)
        soh = C / (C_ref_run + eps)

        # ---- causal slope over past window
        def past_slope(x, W):
            slopes = np.zeros_like(x)
            for t in range(len(x)):
                s = max(0, t-W+1)
                y = x[s:t+1]
                if len(y) < 2:
                    slopes[t] = 0.0
                else:
                    tt = np.arange(len(y))
                    slopes[t] = np.polyfit(tt, y, 1)[0]
            return slopes

        cap_vel = past_slope(C, W)
        cap_deriv = np.diff(C, prepend=C[0])

        # ---- growth rates (causal)
        imp_growth = 100.0 * (Z - Z_ref0) / (Z_ref0 + eps)
        dcr_t = Re  # DCR proxy: per-cycle Re
        dcr_growth = 100.0 * (dcr_t - Re_ref0) / (Re_ref0 + eps)

        imp_growth_log = np.sign(imp_growth) * np.log1p(np.abs(imp_growth))
        dcr_growth_log = np.sign(dcr_growth) * np.log1p(np.abs(dcr_growth))

        # ---- LLI / LAM causal proxy
        lli_proxy = dV / (dC + eps)
        lam_proxy = dRe / (dC + eps)

        # ---- thermal proxies (Arrhenius-like)
        if "temperature_measured_max" in g.columns and "ambient_temp_c" in g.columns:
            Tmax = g["temperature_measured_max"].astype(float).values
            Tamb = g["ambient_temp_c"].astype(float).values
            dT = Tmax - Tamb
            Tref = np.nanmedian(Tmax[:k0])
            alpha = 0.05
            thermal_stress = np.exp(alpha * (Tmax - Tref))
        else:
            dT = np.zeros(len(g))
            thermal_stress = np.zeros(len(g))

        # write back (official formula-based columns)
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

        g["lli"] = lli_proxy
        g["lam"] = lam_proxy

        g["temp_rise"] = dT
        g["thermal_stress"] = thermal_stress

        out.append(g)

    return pd.concat(out, ignore_index=True)


# =============================================================================
# 3. CEEMDAN (kept same as your pipeline)
# =============================================================================
class AdvancedCEEMDAN:
    def __init__(self, noise_mode='adaptive', n_ensembles=150, seed=42):
        self.noise_mode = noise_mode
        self.n_ensembles = n_ensembles
        np.random.seed(seed)

    def analyze_signal_characteristics(self, signal):
        signal_mean = np.mean(signal)
        signal_std = np.std(signal)
        signal_normalized = (signal - signal_mean) / (signal_std + 1e-10)
        signal_diff = np.diff(signal_normalized)
        volatility = np.std(signal_diff)
        skewness = np.abs(np.mean(signal_diff ** 3) / (volatility ** 3 + 1e-10))
        return {'mean': signal_mean, 'std': signal_std, 'normalized': signal_normalized,
                'volatility': volatility, 'skewness': skewness}

    def set_adaptive_noise(self, signal_chars):
        volatility = signal_chars['volatility']
        skewness = signal_chars['skewness']
        base_noise = 0.2 * volatility
        base_noise *= (1 + 0.3 * np.tanh(skewness))
        return np.clip(base_noise, 0.01, 0.3)

    def find_extrema_advanced(self, signal, order=2):
        if len(signal) < 5:
            return np.array([], dtype=int), np.array([], dtype=int)
        maxima_idx = argrelextrema(signal, np.greater, order=order)[0]
        minima_idx = argrelextrema(signal, np.less, order=order)[0]
        return maxima_idx, minima_idx

    def get_smooth_envelope(self, signal, extrema_idx, kind='max'):
        if len(extrema_idx) < 2:
            return np.full_like(signal, np.max(signal) if kind == 'max' else np.min(signal))
        try:
            return CubicSpline(extrema_idx, signal[extrema_idx], bc_type='natural',
                               extrapolate='linear')(np.arange(len(signal)))
        except:
            return np.interp(np.arange(len(signal)), extrema_idx, signal[extrema_idx])

    def sift_advanced(self, signal, max_iterations=200):
        residual = signal.copy()
        for _ in range(max_iterations):
            maxima_idx, minima_idx = self.find_extrema_advanced(residual, order=2)
            if len(maxima_idx) < 2 or len(minima_idx) < 2:
                break
            upper_env = self.get_smooth_envelope(residual, maxima_idx, 'max')
            lower_env = self.get_smooth_envelope(residual, minima_idx, 'min')
            mean_env = (upper_env + lower_env) / 2
            residual_new = residual - mean_env
            sd = np.sum((residual_new - residual) ** 2) / (np.sum(residual ** 2) + 1e-10)
            if sd < 1e-4:
                break
            residual = residual_new
        return residual

    def decompose(self, signal, max_imfs=6):
        signal_chars = self.analyze_signal_characteristics(signal)
        signal_normalized = signal_chars['normalized']
        signal_std = signal_chars['std']

        noise_std = self.set_adaptive_noise(signal_chars) if self.noise_mode == 'adaptive' else 0.2
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
        residual = residual * signal_std + signal_chars['mean']

        while len(imfs) < 6:
            imfs.append(np.zeros_like(signal))
        return imfs, residual


def build_ceemdan(df: pd.DataFrame) -> pd.DataFrame:
    ceemdan = AdvancedCEEMDAN(noise_mode='adaptive', n_ensembles=150, seed=42)
    results = []

    for battery_id, g in df.groupby("battery"):
        g = g.sort_values("cycle_num")
        signal = g["capacity_ahr"].values.astype(float)
        cycles = g["cycle_num"].values.astype(int)

        if len(signal) < 5:
            continue

        imfs, residual = ceemdan.decompose(signal, max_imfs=6)

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

    return pd.DataFrame(results)


# =============================================================================
# 4. Run
# =============================================================================
def main():
    print("=" * 80)
    print("[Pipeline RE] MAT -> causal DB + CEEMDAN")
    print(f"MAT_DIR: {MAT_DIR}")
    print("=" * 80)

    raw_df = build_raw_db_from_mat(MAT_DIR)

    # causal official features
    df_causal = add_physics_features_causal(raw_df, k_ref=10, W=5)

    # keep only columns we need (drop all others)
    KEEP_COLS = [
        # keys
        "battery", "cycle_num",

        # raw used by model / formulas
        "capacity_ahr",
        "ambient_temp_c",
        "voltage_measured_mean",
        "re_ohm_interp",
        "rct_ohm_interp",
        "temperature_measured_max",
        "discharge_time_sec",

        # causal official physics columns
        "soh",
        "capacity_derivative",
        "cap_vel",
        "regen_strength",
        "impedance_sum",
        "impedance_growth",
        "impedance_growth_log",
        "dcr",
        "dcr_growth",
        "dcr_growth_log",
        "lli",
        "lam",
        "temp_rise",
        "thermal_stress",
    ]
    KEEP_COLS = [c for c in KEEP_COLS if c in df_causal.columns]
    df_causal = df_causal[KEEP_COLS].copy()

    df_causal.to_csv(OUT_DB, index=False)
    print(f"[DONE] causal DB saved -> {OUT_DB}")

    # CEEMDAN 그대로 유지(재생성)
    ceemdan_df = build_ceemdan(df_causal)
    ceemdan_df.to_csv(OUT_CEEMDAN, index=False)
    print(f"[DONE] ceemdan saved -> {OUT_CEEMDAN}")

    print("=" * 80)
    print("✅ Finished")
    print("=" * 80)


if __name__ == "__main__":
    main()
