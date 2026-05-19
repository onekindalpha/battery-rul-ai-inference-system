"""
Create C-rate related columns for NASA battery cycle table.

Inputs:
  - battery_training_data_cleaned_final_causal_with_true_crate.csv  (or any cycle-level table that has:
      battery, cycle_num, capacity_ahr, current_min, current_mean, eff_c_rate, discharge_time_sec)

Outputs:
  - battery_training_data_cleaned_final_causal_with_true_crate_v2_expcond_temp0.csv

Definitions (exact):
  - temp_measured_0_c = ambient_temp_c  (this DB's ambient_temp_c is Temperature_measured[0] at discharge start)
  - experiment_condition: label based on temp_measured_0_c (<=10°C -> "저온 4°C", 15~35 -> "실온", else "<x>°C")
  - c_ref_ahr(battery) = median(capacity_ahr of first k_ref cycles with capacity_ahr > 0.2)
  - c_rate_peak = abs(current_min) / c_ref_ahr
  - c_rate_mean = abs(current_mean) / c_ref_ahr
  - eff_c_rate_C = eff_c_rate / c_ref_ahr
  - discharge_current_class: class by abs(current_min) ~ {1A,2A,4A}
"""

import numpy as np
import pandas as pd

IN_PATH = "battery_training_data_cleaned_final_causal_with_true_crate.csv"
OUT_PATH = "battery_training_data_cleaned_final_causal_with_true_crate_v2_expcond_temp0.csv"

K_REF = 10

def compute_c_ref(g: pd.DataFrame) -> float:
    g = g.sort_values("cycle_num")
    g0 = g[g["capacity_ahr"].notna() & (g["capacity_ahr"] > 0.2)].head(K_REF)
    if len(g0) == 0:
        return np.nan
    return float(np.median(g0["capacity_ahr"].values))

def discharge_class(current_min):
    if pd.isna(current_min):
        return "unknown"
    a = abs(float(current_min))
    if 0.7 <= a <= 1.3: return "1A"
    if 1.5 <= a <= 2.5: return "2A"
    if 3.5 <= a <= 4.5: return "4A"
    return "other"

def exp_cond_from_temp0(t):
    if pd.isna(t):
        return "N/A"
    t = float(t)
    if t <= 10.0:
        return "저온 4°C"
    if 15.0 <= t <= 35.0:
        return "실온"
    return f"{t:.1f}°C"

def main():
    df = pd.read_csv(IN_PATH)

    # Explicit alias: start-of-discharge cell temperature (Temperature_measured[0])
    df["temp_measured_0_c"] = df["ambient_temp_c"]

    # UI label (heuristic based on temp_measured_0_c)
    df["experiment_condition"] = df["temp_measured_0_c"].apply(exp_cond_from_temp0)

    # Per-battery reference capacity
    c_ref = df.groupby("battery", sort=False).apply(compute_c_ref).rename("c_ref_ahr")
    df = df.merge(c_ref, left_on="battery", right_index=True, how="left")

    # C-rate metrics (dimensionless)
    df["c_rate_peak"] = np.abs(df["current_min"]) / df["c_ref_ahr"]
    df["c_rate_mean"] = np.abs(df["current_mean"]) / df["c_ref_ahr"]
    df["eff_c_rate_C"] = df["eff_c_rate"] / df["c_ref_ahr"]

    # Current class label
    df["discharge_current_class"] = df["current_min"].apply(discharge_class)

    df.to_csv(OUT_PATH, index=False)
    print("Wrote:", OUT_PATH, "rows:", len(df), "cols:", df.shape[1])

if __name__ == "__main__":
    main()
