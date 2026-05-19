import numpy as np
import pandas as pd

def add_eol_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Requires columns: cycle_num, pred_rul (or pred_rul_cycles), rul_std, true_rul (optional), true_eol (optional)
    out = df.copy()
    if "pred_rul" not in out.columns and "pred_rul_cycles" in out.columns:
        out["pred_rul"] = out["pred_rul_cycles"]
    if "cycle_num" not in out.columns:
        raise ValueError("cycle_num column is required")

    # pred_EOL = t + pred_RUL
    if "pred_rul" in out.columns:
        out["pred_eol"] = out["cycle_num"] + out["pred_rul"]
    # true RUL from true_eol if present
    if "true_eol" in out.columns:
        out["true_rul"] = out["true_eol"] - out["cycle_num"]
    if "true_rul" in out.columns:
        out["resid_rul"] = out.get("pred_rul", np.nan) - out["true_rul"]
    if "rul_std" in out.columns and "resid_rul" in out.columns:
        std = out["rul_std"].replace(0, np.nan)
        out["z_resid"] = out["resid_rul"] / std
    # delta pred EOL
    if "pred_eol" in out.columns:
        out["delta_pred_eol"] = out["pred_eol"].diff()
    return out

def coverage_stats(resid: np.ndarray, sigma: np.ndarray) -> dict:
    # avoid divide by zero, nan-safe
    mask = np.isfinite(resid) & np.isfinite(sigma) & (sigma > 0)
    if mask.sum() == 0:
        return {"cov_1s": None, "cov_2s": None, "avg_sigma": None, "n": 0}
    r = np.abs(resid[mask])
    s = sigma[mask]
    return {
        "cov_1s": float(np.mean(r <= 1.0 * s)),
        "cov_2s": float(np.mean(r <= 2.0 * s)),
        "avg_sigma": float(np.mean(s)),
        "n": int(mask.sum()),
    }

def first_consecutive(condition: np.ndarray, n: int) -> int | None:
    # return index of first position where condition holds for n consecutive points
    if n <= 1:
        idx = np.where(condition)[0]
        return int(idx[0]) if idx.size else None
    run = 0
    for i, ok in enumerate(condition):
        run = run + 1 if ok else 0
        if run >= n:
            return i - n + 1
    return None
