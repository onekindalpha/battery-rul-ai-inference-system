"""
feature_importance_basic.py

Step 2b: basic DB (NASA raw 요약) 기반 피처 중요도 분석 스크립트

- Input : db/battery_training_data_cleaned_final_basic.csv
- Output: 콘솔 출력 + basic_figs/ 에
          - feature importance bar plot (full / cleaned)
          - 공통 중요도 plot
          - 상관관계 히트맵 저장
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import linregress
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------
# 0. Paths
# ---------------------------------------------------------------------
FILE_PATH = Path(__file__).resolve()
ROOT_DIR = FILE_PATH.parent
while not (ROOT_DIR / "db").exists() and ROOT_DIR != ROOT_DIR.parent:
    ROOT_DIR = ROOT_DIR.parent

DB_DIR = ROOT_DIR / "db"
BASIC_DB = DB_DIR / "battery_training_data_cleaned_final_basic.csv"


# ---------------------------------------------------------------------
# 1. EOL / RUL helper (capacity 기반 RUL)
# ---------------------------------------------------------------------
def compute_eol_info(df: pd.DataFrame):
    """
    battery별로:
      - 초기 10개 용량 max -> max_cap
      - eol_cap = 0.8 * max_cap
      - 후반부(cap 70%~끝) linear slope -> vel
    """
    out = {}
    bid_col = "battery"
    cyc_col = "cycle_num"

    for bid, g in df.groupby(bid_col):
        g = g.sort_values(cyc_col)

        if "capacity_ahr" not in g.columns:
            continue

        cap = g["capacity_ahr"].values.astype(float)
        if len(cap) < 5 or np.isnan(cap).all():
            continue

        # 초기 용량
        max_cap = np.nanmax(cap[:10])
        eol_cap = 0.8 * max_cap

        # 후반부 기울기
        n = len(cap)
        tail = cap[int(n * 0.7):]
        cycles = np.arange(len(tail))

        if len(tail) < 2 or np.isnan(tail).all():
            vel = -1e-3
        else:
            tail_f = np.nan_to_num(tail, nan=float(np.nanmean(tail)))
            try:
                vel = linregress(cycles, tail_f).slope
            except ValueError:
                vel = -1e-3
            if abs(vel) < 1e-5:
                vel = -1e-4

        out[str(bid)] = {"eol_cap": float(eol_cap), "vel": float(vel)}

    return out


def safe_compute_rul(capacity, eol_cap, vel):
    """
    (current - eol) / |vel|, [0, 3000] clip
    """
    if np.isnan(capacity) or np.isnan(eol_cap) or np.isnan(vel):
        return 0.0
    return min(max(0.0, (capacity - eol_cap) / max(abs(vel), 1e-5)), 3000.0)


# ---------------------------------------------------------------------
# 공통: fig dir + bar plot + heatmap util
# ---------------------------------------------------------------------
def get_fig_dir():
    fig_dir = Path(__file__).resolve().parent / "basic_figs"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


def save_bar_plot(values, labels, title, filename):
    fig_dir = get_fig_dir()

    values = np.array(values)
    labels = np.array(labels)

    order = np.argsort(values)[::-1]
    sorted_vals = values[order]
    sorted_labels = labels[order]

    plt.figure(figsize=(8, 0.4 * len(sorted_labels) + 1))
    plt.barh(range(len(sorted_vals)), sorted_vals)
    plt.yticks(range(len(sorted_labels)), sorted_labels)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.tight_layout()

    out_path = fig_dir / filename
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Saved] {title} -> {out_path}")


def save_corr_heatmap(corr_df: pd.DataFrame, title: str, filename: str):
    fig_dir = get_fig_dir()

    plt.figure(figsize=(1.0 * len(corr_df.columns), 1.0 * len(corr_df.columns)))
    im = plt.imshow(corr_df.values, vmin=-1, vmax=1, cmap="coolwarm")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    labels = corr_df.columns.tolist()
    plt.xticks(range(len(labels)), labels, rotation=90)
    plt.yticks(range(len(labels)), labels)
    plt.title(title)
    plt.tight_layout()

    out_path = fig_dir / filename
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[Saved] {title} -> {out_path}")


# ---------------------------------------------------------------------
# 한 세트(feat_cols)에 대해 중요도 계산 + 플롯 저장
# ---------------------------------------------------------------------
def run_feature_importance_for_set(df: pd.DataFrame, feat_cols, tag: str):
    print("\n" + "=" * 80)
    print(f"[BASIC] Feature set: {tag}")
    print("사용 피처:", feat_cols)

    X = df[feat_cols].values.astype(np.float32)
    y = df["rul"].values.astype(np.float32)

    # 유효한 row만
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X = X[mask]
    y = y[mask]

    print(f"[Info] Effective samples ({tag}): {len(y)}")
    if len(y) < 20:
        print(f"[Warn] Too few samples for feature importance ({tag}), skip.")
        return None

    # ----- Train / Test split -----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ----- RandomForest 기반 중요도 -----
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    rf_importances = rf.feature_importances_
    idx = np.argsort(rf_importances)[::-1]

    print("\n=== [1] RandomForest Gini Feature Importances (NASA basic) ===")
    for rank, i in enumerate(idx, 1):
        print(f"{rank:2d}. {feat_cols[i]:25s}  {rf_importances[i]:.4f}")

    save_bar_plot(
        values=rf_importances,
        labels=feat_cols,
        title=f"RF Gini Importances (basic, {tag})",
        filename=f"fi_rf_gini_basic_{tag}.png",
    )

    # ----- Permutation Importance (test set) -----
    print("\n[Info] Computing permutation importance on test set...")
    perm = permutation_importance(
        rf,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )
    perm_mean = perm.importances_mean
    p_idx = np.argsort(perm_mean)[::-1]

    print("\n=== [2] Permutation Importances (Test RUL, mean drop in score) ===")
    for rank, i in enumerate(p_idx, 1):
        print(f"{rank:2d}. {feat_cols[i]:25s}  {perm_mean[i]:.4f}")

    save_bar_plot(
        values=perm_mean,
        labels=feat_cols,
        title=f"Permutation Importances (basic, {tag})",
        filename=f"fi_perm_basic_{tag}.png",
    )

    # ----- 상관계수 기반 중요도 -----
    print("\n=== [3] |Pearson correlation| vs RUL ===")
    corr_vals = []
    for j, col in enumerate(feat_cols):
        xj = X[:, j]
        if np.std(xj) < 1e-8 or np.std(y) < 1e-8:
            r = 0.0
        else:
            r = float(np.corrcoef(xj, y)[0, 1])
        corr_vals.append(abs(r))

    corr_vals = np.array(corr_vals)
    c_idx = np.argsort(corr_vals)[::-1]
    for rank, i in enumerate(c_idx, 1):
        print(f"{rank:2d}. {feat_cols[i]:25s}  {corr_vals[i]:.4f}")

    save_bar_plot(
        values=corr_vals,
        labels=feat_cols,
        title=f"|Pearson corr| vs RUL (basic, {tag})",
        filename=f"fi_corr_basic_{tag}.png",
    )

    # 나중에 공통 중요도 계산용으로 반환
    return {
        "feat_cols": feat_cols,
        "rf": rf_importances,
        "perm": perm_mean,
        "corr": corr_vals,
    }


def make_common_importance_plot(metrics: dict, tag: str):
    """
    RF / Perm / Corr 세 벡터를 정규화해서 합친 공통 중요도 스코어 플롯
    """
    if metrics is None:
        return

    feat_cols = metrics["feat_cols"]
    rf = np.array(metrics["rf"], dtype=float)
    perm = np.array(metrics["perm"], dtype=float)
    corr = np.array(metrics["corr"], dtype=float)

    def norm(v):
        s = v.sum()
        if s <= 0:
            return np.zeros_like(v)
        return v / s

    rf_n = norm(rf)
    perm_n = norm(perm)
    corr_n = norm(corr)

    common_score = rf_n + perm_n + corr_n

    idx = np.argsort(common_score)[::-1]
    print("\n=== [4] Common importance score (normalized RF + Perm + |Corr|) ===")
    for rank, i in enumerate(idx, 1):
        print(f"{rank:2d}. {feat_cols[i]:25s}  {common_score[i]:.4f}")

    save_bar_plot(
        values=common_score,
        labels=feat_cols,
        title=f"Common importance (basic, {tag})",
        filename=f"fi_common_basic_{tag}.png",
    )


# ---------------------------------------------------------------------
# 2. Feature importance main logic
# ---------------------------------------------------------------------
def main():
    print("=" * 80)
    print("[Feature Importance] basic DB (NASA raw) 기반 피처 중요도 분석")
    print(f"BASIC_DB: {BASIC_DB}")
    print("=" * 80)

    if not BASIC_DB.exists():
        raise FileNotFoundError(f"basic DB not found: {BASIC_DB}")

    # ----- 데이터 로드 -----
    df = pd.read_csv(BASIC_DB)

    # 숫자형 변환 & 기본 클린
    for col in df.columns:
        if col in ("battery",):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=["capacity_ahr"]).reset_index(drop=True)
    df = df.fillna(0.0)

    # basic DB에서는 thermal proxy 간단하게 추가
    if {"temperature_measured_max", "ambient_temp_c"} <= set(df.columns):
        df["temp_rise"] = df["temperature_measured_max"] - df["ambient_temp_c"]
    else:
        df["temp_rise"] = 0.0

    if "discharge_time_sec" in df.columns:
        df["thermal_stress"] = df["temp_rise"] / (df["discharge_time_sec"] + 1e-9)
    else:
        df["thermal_stress"] = 0.0

    print("[Info] Loaded basic DB:")
    print(df.head())
    print("[Info] Columns:", df.columns.tolist())

    # ----- EOL 정보 & RUL 계산 (capacity 기반) -----
    eol_info = compute_eol_info(df)
    print(f"[Info] EOL info computed for {len(eol_info)} batteries")

    bid_col = "battery"
    cap_col = "capacity_ahr"

    df["rul"] = np.nan
    used_rows = []

    for bid, g in df.groupby(bid_col):
        bid_str = str(bid)
        if bid_str not in eol_info:
            continue

        info = eol_info[bid_str]
        cap = g[cap_col].values.astype(float)

        rul_vals = [
            safe_compute_rul(c, info["eol_cap"], info["vel"])
            for c in cap
        ]
        df.loc[g.index, "rul"] = rul_vals
        used_rows.extend(list(g.index))

    df = df.loc[used_rows].reset_index(drop=True)
    df["rul"] = df["rul"].fillna(0.0)

    print("[Info] RUL computed. RUL stats:")
    print(df["rul"].describe())

    # ----- 피처 선택 (NASA raw 위주) -----
    physics_cols_basic = [
        # capacity / temp
        "capacity_ahr",
        "capacity_mean",
        "ambient_temp_c",
        "temperature_measured_max",
        "temperature_mean",
        "temperature_min",
        "temperature_std",
        "temp_rise",
        "thermal_stress",

        # voltage
        "voltage_measured_mean",
        "voltage_min",
        "voltage_max",
        "voltage_std",

        # impedance / IR
        "re_ohm_interp",
        "rct_ohm_interp",
        "ir_mean",
        "ir_max",

        # operation / current
        "discharge_time_sec",
        "eff_c_rate",
        "current_mean",
        "current_std",
        "current_min",
        "current_max",

        # dynamics
        "dvdt_max_abs",
        "dTdt_max",
    ]

    feat_cols_full = [c for c in physics_cols_basic if c in df.columns]

    # ---- (B) 상관계수/누출 기반 cleaned feature set 만들기 ----
    manual_drop = {"capacity_ahr", "discharge_time_sec"}

    corr = df[feat_cols_full].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    high_corr_drop = {
        col
        for col in upper.columns
        if any(upper[col] > 0.97)
    }

    to_drop = manual_drop | high_corr_drop
    feat_cols_clean = [c for c in feat_cols_full if c not in to_drop]

    print("[Info] dropped (manual + corr):", sorted(to_drop))
    print("[Info] cleaned feature cols:", feat_cols_clean)

    # --- 1) FULL feature set ---
    metrics_full = run_feature_importance_for_set(df, feat_cols_full, tag="full")

    # --- 2) CLEANED feature set ---
    metrics_clean = None
    if len(feat_cols_clean) >= 2:
        metrics_clean = run_feature_importance_for_set(df, feat_cols_clean, tag="cleaned")
    else:
        print("[Warn] cleaned feature set too small, skip.")

    # --- 3) 공통 중요도 플롯 ---
    make_common_importance_plot(metrics_full, "full")
    if metrics_clean is not None:
        make_common_importance_plot(metrics_clean, "cleaned")

    # --- 4) 상관관계 히트맵 (피처 + RUL) ---
    if len(feat_cols_full) > 0:
        corr_full = df[feat_cols_full + ["rul"]].corr()
        save_corr_heatmap(
            corr_full,
            title="Correlation heatmap (basic full)",
            filename="corr_heatmap_basic_full.png",
        )

    if len(feat_cols_clean) >= 2:
        corr_clean = df[feat_cols_clean + ["rul"]].corr()
        save_corr_heatmap(
            corr_clean,
            title="Correlation heatmap (basic cleaned)",
            filename="corr_heatmap_basic_cleaned.png",
        )

    print("\n✅ Done. basic DB에 대해 full / cleaned 중요도 + 공통 중요도 + 히트맵까지 생성 완료.")


if __name__ == "__main__":
    main()
