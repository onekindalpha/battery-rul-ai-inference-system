# (preprocessing)tabular_baseline_rul.py
#
# Cycle-level basic/causal DB 위에서
# - XGBoost
# - MLPRegressor (sklearn)
#
# 간단한 RUL 회귀 베이스라인을 돌려보는 스크립트.
#
# 예시 실행:
#
#   # causal DB + XGBoost + cleaned feature set(기본)
#   python "(preprocessing)tabular_baseline_rul.py" --db causal --model xgb
#
#   # basic DB + MLP + full feature set
#   python "(preprocessing)tabular_baseline_rul.py" --db basic --model mlp --feature-set full
#
#   # causal DB + CEEMDAN feature merge + 둘 다 + cleaned feature + 분석
#   python "(preprocessing)tabular_baseline_rul.py" \
#       --db causal --model both --feature-set cleaned --use-ceemdan --analyze-rul
#
# 필수 의존성:
#   - numpy, pandas, scikit-learn
#   - xgboost (옵션: XGBoost 쓸 때만 필요)
#   - matplotlib (옵션: --analyze-rul 쓸 때만 필요)
#

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

try:
    from xgboost import XGBRegressor
except Exception as e:
    print("[Warn] xgboost import failed, XGBoost baselines will be skipped.")
    print("       Error:", e)
    XGBRegressor = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# ---------------------------------------------------------------------
# 0. Paths
# ---------------------------------------------------------------------
FILE_PATH = Path(__file__).resolve()
ROOT_DIR = FILE_PATH.parent
while not (ROOT_DIR / "db").exists() and ROOT_DIR != ROOT_DIR.parent:
    ROOT_DIR = ROOT_DIR.parent

DB_DIR = ROOT_DIR / "db"
BASIC_DB = DB_DIR / "battery_training_data_cleaned_final_basic.csv"
CAUSAL_DB = DB_DIR / "battery_training_data_cleaned_final_causal.csv"
CEEMDAN_DB = DB_DIR / "ceemdan_all_batteries_advanced_causal.csv"

# ---------------------------------------------------------------------
# 1. EOL / RUL Helpers  (fallback 용)
# ---------------------------------------------------------------------
def compute_eol_info(df: pd.DataFrame):
    """
    train/data.py 의 compute_eol_info 와 동일.

    battery별로:
      - 초기 10개 용량 max -> max_cap
      - eol_cap = 0.8 * max_cap
      - 후반부(70%~끝) 용량 선형 회귀 slope -> vel
    """
    from typing import Dict
    from scipy.stats import linregress

    out: Dict[str, Dict[str, float]] = {}
    bid_col = "battery_id" if "battery_id" in df.columns else "battery"
    cyc_col = "cycle" if "cycle" in df.columns else "cycle_num"

    for bid, g in df.groupby(bid_col):
        g = g.sort_values(cyc_col)

        # 용량 컬럼 찾기
        cap_col = "capacity_ahr" if "capacity_ahr" in g.columns else "Capacity"
        if cap_col not in g.columns:
            continue

        cap = g[cap_col].values.astype(float)
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
    (current_capacity - eol_cap) / |vel|  을 cycles 단위 RUL 로 사용.
    [0, 3000]으로 클립 (train/data.py 와 동일).
    """
    if np.isnan(capacity) or np.isnan(eol_cap) or np.isnan(vel):
        return 0.0
    return min(max(0.0, (capacity - eol_cap) / max(abs(vel), 1e-5)), 3000.0)


def add_rul_column(df: pd.DataFrame, target_col: str = "rul") -> pd.DataFrame:
    """
    df에 RUL(target_col) 컬럼 추가해서 리턴.
    basic/causal 둘 다 같은 정의로 사용.
    """
    bid_col = "battery_id" if "battery_id" in df.columns else "battery"
    cap_col = "capacity_ahr" if "capacity_ahr" in df.columns else "Capacity"

    eol_info = compute_eol_info(df)
    print(f"[Info] EOL info computed for {len(eol_info)} batteries")

    df = df.copy()
    df[target_col] = np.nan
    used_idx = []

    for bid, g in df.groupby(bid_col):
        info = eol_info.get(str(bid))
        if info is None:
            continue

        cap = g[cap_col].values.astype(float)
        rul_vals = [
            safe_compute_rul(c, info["eol_cap"], info["vel"])
            for c in cap
        ]
        df.loc[g.index, target_col] = rul_vals
        used_idx.extend(list(g.index))

    df = df.loc[used_idx].reset_index(drop=True)
    df[target_col] = df[target_col].fillna(0.0)

    print("[Info] RUL stats (newly computed):")
    print(df[target_col].describe())
    return df

def drop_manual_bad_batteries(
    df: pd.DataFrame,
    bad_bids = None,
):
    """
    수동으로 특정 battery id 들을 통째로 제거하는 필터.

    예:
      - 샘플 수 너무 적고(mean_rul ~0) 노이즈만 주는 B0050, B0052
      - (원하면) 도메인이 완전 다른 B0033 등
    """
    if bad_bids is None:
        # 기본으로 항상 빼고 싶은 애들
        bad_bids = ["B0050", "B0052"]

    bid_col = "battery_id" if "battery_id" in df.columns else "battery"

    bad_bids = set(str(b) for b in bad_bids)
    mask_bad = df[bid_col].astype(str).isin(bad_bids)

    before = len(df)
    df_f = df[~mask_bad].reset_index(drop=True)
    after = len(df_f)

    dropped = df.loc[mask_bad, bid_col].astype(str).unique().tolist()

    if dropped:
        print(
            f"[Filter] Manually dropped batteries {dropped} "
            f"({before} -> {after} rows)"
        )
    else:
        print("[Filter] No manual bad batteries found in this DB.")

    return df_f, dropped

def drop_constant_rul_batteries(
    df: pd.DataFrame,
    target_col: str,
    min_range: float = 10.0,
    min_std: float = 5.0,
    min_samples: int = 20,
):
    """
    배터리별로 RUL 분포가 거의 안 변하는(사실상 상수인) 배터리들을 통째로 제거.

    기준:
      - 배터리별 n >= min_samples 이고
      - (max_rul - min_rul) < min_range  또는  std_rul < min_std

    이렇게 생긴 애들이 지금 B0028, B0056, B0053 류 (mean_true_rul ~ 0)라고 보면 됨.
    """
    bid_col = "battery_id" if "battery_id" in df.columns else "battery"

    g = df.groupby(bid_col)[target_col]
    stats = g.agg(
        n="count",
        mean="mean",
        std="std",
        min="min",
        max="max",
    ).reset_index()
    stats["range"] = stats["max"] - stats["min"]

    mask_bad = (
        (stats["n"] >= min_samples)
        & ((stats["range"] < min_range) | (stats["std"] < min_std))
    )
    bad_bids = stats.loc[mask_bad, bid_col].astype(str).tolist()

    if not bad_bids:
        print("[Filter] No constant-RUL-like batteries found.")
        return df, bad_bids

    print(
        "[Filter] Constant-RUL-like batteries to drop "
        f"(range < {min_range} or std < {min_std}, n >= {min_samples}): {bad_bids}"
    )

    before = len(df)
    df_f = df[~df[bid_col].astype(str).isin(bad_bids)].reset_index(drop=True)
    after = len(df_f)

    print(
        f"[Filter] Dropped {before - after} rows from {len(bad_bids)} batteries "
        f"({before} -> {after})"
    )
    return df_f, bad_bids


# ---------------------------------------------------------------------
# 2. Feature selection (full / cleaned)
# ---------------------------------------------------------------------
def select_feature_columns(df: pd.DataFrame, db_type: str, feature_set: str):
    """
    - numeric 컬럼 중에서
      * ID / target / cycle_life 등 제거
      * manual_drop(라벨 누출 의심) 제거
      * std ~0 인 상수 컬럼 제거
    - feature_set == "cleaned" 인 경우:
      * 상관계수 |corr| > 0.97 인 컬럼은 한쪽만 남김
    """
    bid_col = "battery_id" if "battery_id" in df.columns else "battery"

    id_cols = {
        bid_col,
        "battery",
        "battery_id",
        "cycle",
        "cycle_num",
    }
    target_cols = {"rul", "rul_cycles", "rul_norm", "cycle_life"}
    manual_drop = {
        # 직접적인 라벨 누출 강한 애들 (필요하면 빼고 써도 됨)
        "capacity_ahr",
        "discharge_time_sec",
    }

    exclude = id_cols | target_cols | manual_drop

    # numeric 컬럼만
    num_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]

    # 상수 컬럼 제거
    stds = df[num_cols].std()
    num_cols = [c for c in num_cols if stds[c] > 1e-8]

    if feature_set == "full":
        print(f"[Info] Using FULL feature set ({db_type}): {len(num_cols)} cols")
        print(num_cols)
        return num_cols

    # cleaned: high-corr drop
    corr = df[num_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    high_corr_drop = {
        col
        for col in upper.columns
        if any(upper[col] > 0.97)
    }

    cleaned = [c for c in num_cols if c not in high_corr_drop]
    print(f"[Info] Using CLEANED feature set ({db_type}): {len(cleaned)} cols")
    print("[Info] dropped (corr>0.97):", sorted(high_corr_drop))
    print("[Info] final feature cols:", cleaned)
    return cleaned


# ---------------------------------------------------------------------
# 3. DB 로드 + 공통 전처리
# ---------------------------------------------------------------------
def load_db(db_type: str, use_ceemdan: bool = False):
    """
    db_type:
      - 'basic'  : 기존 basic DB만 사용
      - 'causal' : 기존 causal DB (+ 옵션으로 CEEMDAN merge)
    use_ceemdan:
      - True  이면 causal DB에 CEEMDAN advanced feature를 merge
    """
    if db_type == "basic":
        if use_ceemdan:
            raise ValueError("basic DB에는 CEEMDAN을 붙이지 않는 걸로 가정했음.")
        path = BASIC_DB
        if not path.exists():
            raise FileNotFoundError(path)

        print("=" * 80)
        print(f"[Load] DB type=basic, path={path}")
        print("=" * 80)

        df = pd.read_csv(path)

    elif db_type == "causal":
        feat_path = CAUSAL_DB
        if not feat_path.exists():
            raise FileNotFoundError(feat_path)

        print("=" * 80)
        print(f"[Load] DB type=causal, path={feat_path}")
        if use_ceemdan:
            print(f"[Load]   + CEEMDAN features: {CEEMDAN_DB}")
        print("=" * 80)

        feat = pd.read_csv(feat_path)

        if use_ceemdan:
            if not CEEMDAN_DB.exists():
                raise FileNotFoundError(CEEMDAN_DB)
            ceem = pd.read_csv(CEEMDAN_DB)

            # CEEMDAN Capacity는 label leak 방지 차원에서 제거
            ceem = ceem.drop(columns=[c for c in ["Capacity"] if c in ceem.columns],
                             errors="ignore")

            merge_keys = ["battery", "cycle_num"]
            missing = set(merge_keys) - set(feat.columns)
            if missing:
                raise KeyError(f"causal DB에 merge key {missing} 이(가) 없음")
            missing = set(merge_keys) - set(ceem.columns)
            if missing:
                raise KeyError(f"CEEMDAN DB에 merge key {missing} 이(가) 없음")

            df = pd.merge(feat, ceem, on=merge_keys, how="left")
            print(f"[Info] After CEEMDAN merge: shape={df.shape}")
        else:
            df = feat

    else:
        raise ValueError(f"Unknown db_type: {db_type}")

    # 숫자형 캐스팅 + 기본 클린
    for col in df.columns:
        if col in ("battery", "battery_id"):
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=["capacity_ahr"]).reset_index(drop=True)
    df = df.fillna(0.0)

    print("[Info] Loaded DB head:")
    print(df.head())
    print("[Info] Columns:", df.columns.tolist())

    # basic DB용 thermal_stress 보정
    if (
        "temperature_measured_max" in df.columns
        and "ambient_temp_c" in df.columns
        and "discharge_time_sec" in df.columns
        and "thermal_stress" not in df.columns
    ):
        temp_rise = df["temperature_measured_max"] - df["ambient_temp_c"]
        df["temp_rise"] = temp_rise
        df["thermal_stress"] = temp_rise / (df["discharge_time_sec"] + 1e-9)

    # ---- RUL 타깃 선택 ----
    target_col = None
    if "rul_cycles" in df.columns:
        target_col = "rul_cycles"       # ← CSV에 있으면 이걸 최우선으로 사용
    elif "rul_norm" in df.columns:
        target_col = "rul_norm"
    elif "rul" in df.columns:
        target_col = "rul"

    if target_col is None:
        print("[Info] RUL column not found in DB. Computing RUL via add_rul_column().")
        df = add_rul_column(df, target_col="rul")
        target_col = "rul"
    else:
        print(f"[Info] Using existing RUL column: {target_col}")
        df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    # NaN RUL 제거
    before = len(df)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    after = len(df)
    print(f"[Info] Dropped {before - after} rows with NaN {target_col}")
    print("[Info] RUL stats (before constant-RUL filter):")
    print(df[target_col].describe())

    # constant-RUL 배터리 제거
    df, dropped_bids = drop_constant_rul_batteries(
        df,
        target_col=target_col,
        min_range=10.0,
        min_std=5.0,
        min_samples=20,
    )
    if dropped_bids:
        print(f"[Info] Dropped constant-RUL batteries: {dropped_bids}")

    print("[Info] RUL stats (after constant-RUL filter):")
    print(df[target_col].describe())
    # 🔽🔽🔽 여기 추가 🔽🔽🔽
    # 수동으로 문제 배터리 제거 (B0050, B0052, 원하면 B0033 등)
    df, dropped_manual = drop_manual_bad_batteries(
        df,
        bad_bids=["B0050", "B0052"]  # B0033도 빼고 싶으면 여기 추가
    )
    if dropped_manual:
        print(f"[Info] Dropped manual bad batteries: {dropped_manual}")

    print("[Info] RUL stats (after manual battery filter):")
    print(df[target_col].describe())
    # 🔼🔼🔼 여기까지 추가 🔼🔼🔼
    return df, target_col

# ---------------------------------------------------------------------
# 4. Baseline models
# ---------------------------------------------------------------------
def run_xgb_baseline(X, y, groups, feature_names):
    if XGBRegressor is None:
        print("[Warn] xgboost is not installed. Skip XGB baseline.")
        return

    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)
    gkf = GroupKFold(n_splits=n_splits)

    rmses, maes, r2s = [], [], []

    print("\n" + "=" * 80)
    print("[XGBoost] GroupKFold cross-validation")
    print("=" * 80)

    fold = 0
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        model = XGBRegressor(
            n_estimators=600,
            learning_rate=0.03,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="reg:squarederror",
            tree_method="hist",
            random_state=42,
        )
        model.fit(X_tr, y_tr)

        pred = model.predict(X_te)
        rmse = np.sqrt(mean_squared_error(y_te, pred))
        mae = mean_absolute_error(y_te, pred)
        r2 = r2_score(y_te, pred)

        rmses.append(rmse)
        maes.append(mae)
        r2s.append(r2)

        print(f"[Fold {fold}] RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}")

    if fold == 0:
        print("[Warn] Not enough folds for XGB.")
        return

    print("\n[XGB] CV summary:")
    print(f"  RMSE: {np.mean(rmses):.3f} ± {np.std(rmses):.3f}")
    print(f"  MAE : {np.mean(maes):.3f} ± {np.std(maes):.3f}")
    print(f"  R2  : {np.mean(r2s):.3f} ± {np.std(r2s):.3f}")


def run_mlp_baseline(X, y, groups, feature_names):
    n_groups = len(np.unique(groups))
    n_splits = min(5, n_groups)
    gkf = GroupKFold(n_splits=n_splits)

    rmses, maes, r2s = [], [], []
    all_y_true, all_y_pred, all_groups = [], [], []

    print("\n" + "=" * 80)
    print("[MLP] GroupKFold cross-validation")
    print("=" * 80)

    fold = 0
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(X, y, groups), 1):
        X_tr, X_te = X[tr_idx], X[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        g_te       = groups[te_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        mlp = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=500,
            random_state=42,
            early_stopping=True,
            n_iter_no_change=20,
        )
        mlp.fit(X_tr_s, y_tr)

        pred = mlp.predict(X_te_s)

        rmse = np.sqrt(mean_squared_error(y_te, pred))
        mae = mean_absolute_error(y_te, pred)
        r2 = r2_score(y_te, pred)

        rmses.append(rmse)
        maes.append(mae)
        r2s.append(r2)

        all_y_true.append(y_te)
        all_y_pred.append(pred)
        all_groups.append(g_te)

        print(f"[Fold {fold}] RMSE={rmse:.3f}, MAE={mae:.3f}, R2={r2:.3f}")

    if fold == 0:
        print("[Warn] Not enough folds for MLP.")
        return

    # 1) fold 평균
    print("\n[MLP] CV summary (mean of folds):")
    print(f"  RMSE: {np.mean(rmses):.3f} ± {np.std(rmses):.3f}")
    print(f"  MAE : {np.mean(maes):.3f} ± {np.std(maes):.3f}")
    print(f"  R2  : {np.mean(r2s):.3f} ± {np.std(r2s):.3f}")

    # 2) GLOBAL metric
    y_true_all   = np.concatenate(all_y_true)
    y_pred_all   = np.concatenate(all_y_pred)
    groups_all   = np.concatenate(all_groups).astype(str)

    global_rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
    global_mae  = mean_absolute_error(y_true_all, y_pred_all)
    global_r2   = r2_score(y_true_all, y_pred_all)

    print("\n[MLP] GLOBAL metrics (all folds concatenated):")
    print(f"  RMSE: {global_rmse:.3f}")
    print(f"  MAE : {global_mae:.3f}")
    print(f"  R2  : {global_r2:.3f}")

    # 3) Dummy mean baseline
    dummy_pred = np.full_like(y_true_all, fill_value=y_true_all.mean(), dtype=float)
    dummy_rmse = np.sqrt(mean_squared_error(y_true_all, dummy_pred))
    dummy_mae  = mean_absolute_error(y_true_all, dummy_pred)

    print("\n[MLP] Dummy-mean baseline (GLOBAL):")
    print(f"  Dummy RMSE: {dummy_rmse:.3f}")
    print(f"  Dummy MAE : {dummy_mae:.3f}")

    if dummy_rmse > 0:
        rmse_improve = 100.0 * (1.0 - global_rmse / dummy_rmse)
    else:
        rmse_improve = 0.0

    if dummy_mae > 0:
        mae_improve = 100.0 * (1.0 - global_mae / dummy_mae)
    else:
        mae_improve = 0.0

    print("\n[MLP] Improvement over dummy-mean (GLOBAL):")
    print(f"  RMSE improvement: {rmse_improve:.2f}%  "
          f"(model={global_rmse:.3f}, dummy={dummy_rmse:.3f})")
    print(f"  MAE  improvement: {mae_improve:.2f}%  "
          f"(model={global_mae:.3f}, dummy={dummy_mae:.3f})")

    # ====== RUL <= 50 구간 성능 ======
    mask_short = y_true_all <= 50
    if mask_short.any():
        rmse_short = np.sqrt(mean_squared_error(
            y_true_all[mask_short], y_pred_all[mask_short]
        ))
        mae_short  = mean_absolute_error(
            y_true_all[mask_short], y_pred_all[mask_short]
        )
        print(f"\n[MLP] RUL <= 50 subset: "
              f"RMSE={rmse_short:.3f}, MAE={mae_short:.3f}, n={mask_short.sum()}")
    else:
        print("\n[MLP] RUL <= 50 subset: no samples.")

    # 4) 배터리별 에러
    from collections import defaultdict

    per_batt = defaultdict(list)
    for gid, yt, yp in zip(groups_all, y_true_all, y_pred_all):
        per_batt[gid].append((yt, yp))

    stats = []
    for bid, pairs in per_batt.items():
        arr = np.array(pairs)
        yt_b = arr[:, 0]
        yp_b = arr[:, 1]
        mae_b  = mean_absolute_error(yt_b, yp_b)
        rmse_b = np.sqrt(mean_squared_error(yt_b, yp_b))
        mean_rul = yt_b.mean()
        stats.append((bid, len(yt_b), mae_b, rmse_b, mean_rul))

    stats_sorted = sorted(stats, key=lambda x: x[2], reverse=True)

    print("\n[MLP] Per-battery error (top 5 by MAE):")
    print("  bid    n_samples   MAE    RMSE   mean_true_rul")
    for bid, n, mae_b, rmse_b, m_rul in stats_sorted[:5]:
        print(f"  {bid:6s} {n:9d}  {mae_b:6.2f} {rmse_b:7.2f}  {m_rul:7.2f}")

    # 분석용 전체 결과 반환
    return {
        "y_true_all": y_true_all,
        "y_pred_all": y_pred_all,
        "groups_all": groups_all,
    }

# ---------------------------------------------------------------------
# 4-1. 분석 유틸
# ---------------------------------------------------------------------
def print_rul_binned_metrics(y_true_all, y_pred_all,
                             bins=(0, 50, 100, 200, 500, np.inf)):
    print("\n[MLP] RUL-binned metrics:")
    bins = list(bins)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_true_all > lo) & (y_true_all <= hi)
        if not mask.any():
            continue
        rmse = np.sqrt(mean_squared_error(y_true_all[mask], y_pred_all[mask]))
        mae  = mean_absolute_error(y_true_all[mask], y_pred_all[mask])
        print(
            f"  ({lo:6.1f}, {hi:6.1f}] : "
            f"n={mask.sum():5d}, RMSE={rmse:8.3f}, MAE={mae:8.3f}"
        )


def analyze_rul_predictions(y_true_all, y_pred_all, groups_all,
                            output_dir: Path,
                            top_k: int = 5):
    """
    - RUL 구간별 metric 출력
    - per-battery MAE 기준 worst top_k에 대해 true vs pred scatter png 저장
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) 구간별 metric
    print_rul_binned_metrics(y_true_all, y_pred_all)

    # 2) 배터리별 stats
    from collections import defaultdict
    per_batt = defaultdict(list)
    for gid, yt, yp in zip(groups_all, y_true_all, y_pred_all):
        per_batt[gid].append((yt, yp))

    stats = []
    for bid, pairs in per_batt.items():
        arr = np.array(pairs)
        yt_b = arr[:, 0]
        yp_b = arr[:, 1]
        mae_b  = mean_absolute_error(yt_b, yp_b)
        rmse_b = np.sqrt(mean_squared_error(yt_b, yp_b))
        mean_rul = yt_b.mean()
        stats.append((bid, len(yt_b), mae_b, rmse_b, mean_rul))

    stats_sorted = sorted(stats, key=lambda x: x[2], reverse=True)

    print("\n[Analyze] Per-battery stats (sorted by MAE desc, top 10):")
    print("  bid    n_samples   MAE    RMSE   mean_true_rul")
    for bid, n, mae_b, rmse_b, m_rul in stats_sorted[:10]:
        print(f"  {bid:6s} {n:9d}  {mae_b:6.2f} {rmse_b:7.2f}  {m_rul:7.2f}")

    if plt is None:
        print("\n[Analyze] matplotlib 이 없어서 플롯은 건너뜀 (pip install matplotlib).")
        return

    # 3) worst top_k 플롯
    print(f"\n[Analyze] Saving scatter plots for worst {top_k} batteries into {output_dir}")
    for bid, n, mae_b, rmse_b, m_rul in stats_sorted[:top_k]:
        arr = np.array(per_batt[bid])
        yt_b = arr[:, 0]
        yp_b = arr[:, 1]

        plt.figure()
        plt.scatter(yt_b, yp_b, alpha=0.6)
        lo = min(yt_b.min(), yp_b.min())
        hi = max(yt_b.max(), yp_b.max())
        plt.plot([lo, hi], [lo, hi], linestyle="--")
        plt.xlabel("True RUL")
        plt.ylabel("Predicted RUL")
        plt.title(f"Battery {bid} (n={n}, MAE={mae_b:.2f}, RMSE={rmse_b:.2f})")
        plt.tight_layout()
        plt.savefig(output_dir / f"scatter_true_vs_pred_{bid}.png", dpi=200)
        plt.close()

# ---------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Tabular RUL baselines on basic/causal DB")
    parser.add_argument("--db", choices=["basic", "causal"], default="causal",
                        help="어느 DB를 쓸지 (basic / causal)")
    parser.add_argument("--model", choices=["xgb", "mlp", "both"], default="both",
                        help="어떤 모델을 돌릴지")
    parser.add_argument("--feature-set", choices=["full", "cleaned"], default="cleaned",
                        help="feature_importance에서처럼 full / cleaned 선택")
    parser.add_argument("--use-ceemdan", action="store_true",
                        help="causal DB에 CEEMDAN feature merge")
    parser.add_argument("--analyze-rul", action="store_true",
                        help="MLP RUL 예측에 대해 구간별/배터리별 분석 및 플롯 저장")
    # ★ 여기 추가
    parser.add_argument(
    "--clip-rul-max",            # ← 이 줄만 옵션 문자열
    type=float,
    default=None,
    help="설정하면 RUL 타깃을 y = min(y, clip_rul_max)로 클리핑 (예: 150)",
)

    args = parser.parse_args()

    # 1) DB 로드 + RUL(target) 결정
    df, target_col = load_db(args.db, use_ceemdan=args.use_ceemdan)
    bid_col = "battery_id" if "battery_id" in df.columns else "battery"

    # 2) feature set 선택
    feature_cols = select_feature_columns(df, db_type=args.db, feature_set=args.feature_set)

    # 3) numpy array로 변환
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32)
    groups = df[bid_col].astype(str).values  # 배터리 단위 그룹
    # ★ 선택적 클리핑
    if args.clip_rul_max is not None:
        H = float(args.clip_rul_max)
        print(f"[Info] Clipping RUL target: y = min(y, {H})")
        print("[Info] RUL stats BEFORE clip:")
        print(pd.Series(y).describe())

        y = np.minimum(y, H).astype(np.float32)

        print("[Info] RUL stats AFTER clip:")
        print(pd.Series(y).describe())
    print("=" * 80)
    print(f"[Info] DB={args.db}, feature_set={args.feature_set}, target={target_col}")
    print(f"[Info] X shape={X.shape}, y shape={y.shape}, num_batteries={len(np.unique(groups))}")
    print(f"[Info] feature_cols ({len(feature_cols)}): {feature_cols}")
    print("=" * 80)

    # 4) 모델별 GroupKFold CV 실행
    if args.model in ("xgb", "both"):
        run_xgb_baseline(X, y, groups, feature_cols)

    mlp_result = None
    if args.model in ("mlp", "both"):
        mlp_result = run_mlp_baseline(X, y, groups, feature_cols)

    # 분석 호출 (플래그가 켜져 있을 때만)
    if args.analyze_rul and mlp_result is not None:
        out_dir = FILE_PATH.parent / "rul_analysis_plots_nasa"
        analyze_rul_predictions(
            y_true_all=mlp_result["y_true_all"],
            y_pred_all=mlp_result["y_pred_all"],
            groups_all=mlp_result["groups_all"],
            output_dir=out_dir,
        )


if __name__ == "__main__":
    main()

# # 돌리는법
# python "(preprocessing)tabular_baseline_rul.py" \
#   --db causal --model both --feature-set full --use-ceemdan --analyze-rul

# 롱 호라이즌 영향 제거 실험
# python "(preprocessing)tabular_baseline_rul.py" \
#   --db causal --model both --feature-set full --use-ceemdan --analyze-rul \
#   --clip-rul-max 150


# train에서 제외: B0052, B0050 (+ constant-RUL 애들은 이미 코드에서 필터됨)
# meta-train에는 빼고 OOD/meta-test로: B0033
# 나머지 (5,6,7,18 포함) → baseline / Transformer / BMAML 학습에 계속 사용 ✅