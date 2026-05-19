"""
feature_importance_causal_cacle.py

CACLE causal DB 기반 물리 피처 중요도 분석 스크립트

- Input : db/cacle_battery_training_data_cleaned_final_causal.csv
- Output: 콘솔 출력 + cacle_causal_figs/ 에
          - feature importance bar plot (full / cleaned)
          - 공통 중요도 plot
          - 상관관계 히트맵 저장
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------
# 0. Paths & target 설정
# ---------------------------------------------------------------------
FILE_PATH = Path(__file__).resolve()
ROOT_DIR = FILE_PATH.parent
while not (ROOT_DIR / "db").exists() and ROOT_DIR != ROOT_DIR.parent:
    ROOT_DIR = ROOT_DIR.parent

DB_DIR = ROOT_DIR / "db"
CAUSAL_DB = DB_DIR / "cacle_battery_training_data_cleaned_final_causal.csv"

# 파이프라인에서 만든 RUL 컬럼 사용
#   - "rul_cycles" : 사이클 단위 RUL
#   - "rul_norm"   : 정규화 RUL (0~1)
# 필요하면 여기서 "soh"나 "rul_norm"으로 바꿔서 실험 가능
TARGET_COL = "rul_cycles"


# ---------------------------------------------------------------------
# 공통: fig dir / bar plot / heatmap util
# ---------------------------------------------------------------------
def get_fig_dir():
    fig_dir = Path(__file__).resolve().parent / "cacle_causal_figs"
    fig_dir.mkdir(exist_ok=True)
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
# + 세 가지 중요도 벡터를 리턴 (공통 스코어용)
# ---------------------------------------------------------------------
def run_feature_importance_for_set(df: pd.DataFrame, feat_cols, tag: str):
    print("\n" + "=" * 80)
    print(f"[CAUSAL-CACLE] Feature set: {tag}")
    print("사용 피처:", feat_cols)

    X = df[feat_cols].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)

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

    print("\n=== [1] RandomForest Gini Feature Importances ===")
    for rank, i in enumerate(idx, 1):
        print(f"{rank:2d}. {feat_cols[i]:25s}  {rf_importances[i]:.4f}")

    # Plot
    save_bar_plot(
        values=rf_importances,
        labels=feat_cols,
        title=f"RF Gini Importances (CACLE causal, {tag})",
        filename=f"fi_rf_gini_cacle_causal_{tag}.png",
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
        title=f"Permutation Importances (CACLE causal, {tag})",
        filename=f"fi_perm_cacle_causal_{tag}.png",
    )

    # ----- 상관계수 기반 중요도 -----
    print("\n=== [3] |Pearson correlation| vs target ===")
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
        title=f"|Pearson corr| vs target (CACLE causal, {tag})",
        filename=f"fi_corr_cacle_causal_{tag}.png",
    )

    # metrics를 리턴해서 공통 스코어 계산에 사용
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
        title=f"Common importance (CACLE causal, {tag})",
        filename=f"fi_common_cacle_causal_{tag}.png",
    )


# ---------------------------------------------------------------------
# 2. Feature importance main logic
# ---------------------------------------------------------------------
def main():
    print("=" * 80)
    print("[Feature Importance] CACLE causal DB 기반 피처 중요도 분석")
    print(f"CAUSAL_DB: {CAUSAL_DB}")
    print("=" * 80)

    if not CAUSAL_DB.exists():
        raise FileNotFoundError(f"causal DB not found: {CAUSAL_DB}")

    # ----- 데이터 로드 -----
    df = pd.read_csv(CAUSAL_DB)

    # 숫자형 변환 & 기본 클린
    for col in df.columns:
        if col in ("battery",):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=["capacity_ahr"]).reset_index(drop=True)

    # target 기준으로 유효 row만 사용
    if TARGET_COL not in df.columns:
        raise KeyError(f"TARGET_COL {TARGET_COL} not in causal DB columns.")

    df = df[np.isfinite(df[TARGET_COL])].reset_index(drop=True)

    print("[Info] Loaded CACLE causal DB:")
    print(df.head())
    print("[Info] Columns:", df.columns.tolist())
    print(f"[Info] Target stats ({TARGET_COL}):")
    print(df[TARGET_COL].describe())

    # ----- 피처 선택 (물리 피처 위주) -----
    physics_cols = [
        # capacity / dynamics
        "capacity_ahr",
        "capacity_mean",
        "soh",
        "capacity_derivative",
        "cap_vel",
        "regen_strength",

        # temperature / thermal (CACLE에는 보통 NaN이지만, 있으면 사용)
        "ambient_temp_c",
        "temperature_measured_max",
        "temperature_mean",
        "temperature_min",
        "temperature_std",
        "temp_rise_cycle",
        "temp_rise",
        "thermal_stress",

        # voltage
        "voltage_measured_mean",
        "voltage_min",
        "voltage_max",
        "voltage_std",
        "v_dod_10",
        "v_dod_50",
        "v_dod_90",

        # impedance / IR
        "re_ohm_interp",
        "rct_ohm_interp",
        "ir_mean",
        "ir_max",
        "impedance_sum",
        "impedance_growth",
        "dcr",
        "dcr_growth",

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

    # 실제로 있는 컬럼 + 전부 NaN이 아닌 컬럼만 사용
    feat_cols_full = []
    dropped_all_nan = []
    for c in physics_cols:
        if c not in df.columns:
            continue
        vals = df[c].values.astype(float)
        if np.isfinite(vals).sum() == 0:
            dropped_all_nan.append(c)
            continue
        feat_cols_full.append(c)

    print("[Info] dropped all-NaN/non-finite cols:", dropped_all_nan)
    print("[Info] full feature cols:", feat_cols_full)

    # ---- (B) 상관계수/누출 기반 cleaned feature set 만들기 ----
    manual_drop = {"capacity_ahr", "discharge_time_sec"}

    if len(feat_cols_full) > 0:
        corr_mat = df[feat_cols_full].corr().abs()
        upper = corr_mat.where(np.triu(np.ones(corr_mat.shape), k=1).astype(bool))

        high_corr_drop = {
            col
            for col in upper.columns
            if any(upper[col] > 0.97)
        }
    else:
        high_corr_drop = set()

    to_drop = manual_drop | high_corr_drop
    feat_cols_clean = [c for c in feat_cols_full if c not in to_drop]

    print("[Info] dropped (manual + corr):", sorted(to_drop))
    print("[Info] cleaned feature cols:", feat_cols_clean)

    # --- 1) FULL feature set ---
    metrics_full = None
    if len(feat_cols_full) >= 2:
        metrics_full = run_feature_importance_for_set(df, feat_cols_full, tag="full")
    else:
        print("[Warn] full feature set too small, skip.")

    # --- 2) CLEANED feature set ---
    metrics_clean = None
    if len(feat_cols_clean) >= 2:
        metrics_clean = run_feature_importance_for_set(df, feat_cols_clean, tag="cleaned")
    else:
        print("[Warn] cleaned feature set too small, skip.")

    # --- 3) 공통 중요도 플롯 ---
    if metrics_full is not None:
        make_common_importance_plot(metrics_full, "full")
    if metrics_clean is not None:
        make_common_importance_plot(metrics_clean, "cleaned")

    # --- 4) 상관관계 히트맵 (피처 + target) ---
    if len(feat_cols_full) > 0:
        corr_full = df[feat_cols_full + [TARGET_COL]].corr()
        save_corr_heatmap(
            corr_full,
            title=f"Correlation heatmap (CACLE causal full, target={TARGET_COL})",
            filename="corr_heatmap_cacle_causal_full.png",
        )

    if len(feat_cols_clean) >= 2:
        corr_clean = df[feat_cols_clean + [TARGET_COL]].corr()
        save_corr_heatmap(
            corr_clean,
            title=f"Correlation heatmap (CACLE causal cleaned, target={TARGET_COL})",
            filename="corr_heatmap_cacle_causal_cleaned.png",
        )

    print("\n✅ Done. CACLE causal DB에 대해 full / cleaned 중요도 + 공통 중요도 + 히트맵까지 생성 완료.")


if __name__ == "__main__":
    main()
