import os
from pathlib import Path
from typing import Dict, Optional, Any, List

import numpy as np
import pandas as pd
import warnings  # <<< 추가

# SciPy skew/kurtosis precision warning 무시 (데이터가 거의 동일할 때 자주 뜸)
warnings.filterwarnings(
    "ignore",
    message="Precision loss occurred in moment calculation due to catastrophic cancellation",
    category=RuntimeWarning,
)

from scipy.stats import skew, kurtosis, linregress, entropy
from scipy.signal import find_peaks  

import torch
import torch.nn.functional as F  # physics_loss에 F.softplus 사용
from torch.utils.data import Dataset

from deep_learning.core_2.config_2 import Config
from deep_learning.core_2.scalers_2 import scale_rul_array

# --------------------------
# Project Root & Config
# --------------------------
FILE_PATH = Path(__file__).resolve()
ROOT_DIR = FILE_PATH.parent
while not (ROOT_DIR / "db").exists() and ROOT_DIR != ROOT_DIR.parent:
    ROOT_DIR = ROOT_DIR.parent
if not (ROOT_DIR / "db").exists():
    ROOT_DIR = FILE_PATH.parent.parent.parent

cfg = Config()


# --------------------------
# [v6 이식] Robust Data Loader
# --------------------------
def load_and_prepare_data(cfg: Config) -> pd.DataFrame:
    """
    v6 스타일의 강력한 데이터 로딩 및 컬럼 보정 로직 적용 (핵심 피처만 남김)
    """
    db = ROOT_DIR / "db"
    
    # 파일 경로 설정 (환경 변수 우선)
    features_path = Path(os.environ.get("BATTERY_DATA_PATH", str(db / "battery_training_data_cleaned_final.csv")))
    ceemdan_path = db / "ceemdan_all_batteries_advanced.csv"

    if not features_path.exists():
        raise FileNotFoundError(f"[Error] Feature file not found: {features_path}")
    if not ceemdan_path.exists():
        raise FileNotFoundError(f"[Error] CEEMDAN file not found: {ceemdan_path}")

    features = pd.read_csv(features_path)
    ceemdan = pd.read_csv(ceemdan_path)

    # [v6 Logic] 컬럼명 자동 보정 (Robust Renaming)
    for df in [features, ceemdan]:
        cols = df.columns
        if 'battery' in cols and 'battery_id' not in cols:
            df.rename(columns={'battery': 'battery_id'}, inplace=True)
        if 'cycle_num' in cols and 'cycle' not in cols:
            df.rename(columns={'cycle_num': 'cycle'}, inplace=True)
        elif 'cycle_index' in cols and 'cycle' not in cols:
            df.rename(columns={'cycle_index': 'cycle'}, inplace=True)

    # [v6 Logic] 중복 컬럼 처리 (Capacity 등)
    if "Capacity" in ceemdan.columns and "capacity_ahr" in features.columns:
        ceemdan = ceemdan.drop(columns=["Capacity"], errors='ignore')
    elif "Capacity" in ceemdan.columns and "Capacity" in features.columns:
        ceemdan = ceemdan.drop(columns=["Capacity"], errors='ignore')

    # 병합
    m = pd.merge(features, ceemdan, on=["battery_id", "cycle"], how="left")
    
    # 숫자형 변환 및 결측치 처리
    for col in m.columns:
        if col in ("battery_id", "battery"):
            continue
        m[col] = pd.to_numeric(m[col], errors="coerce")

    m.replace([np.inf, -np.inf], np.nan, inplace=True)
    num_cols = m.select_dtypes(include=[np.number]).columns
    
    # 간단한 채우기
    m[num_cols] = m[num_cols].fillna(0.0)

    # Global Clipping (Config 설정 따름)
    clip_min = getattr(cfg, "clip_min", -1e6)
    clip_max = getattr(cfg, "clip_max", 1e6)
    m[num_cols] = m[num_cols].clip(clip_min, clip_max)

    # [CRITICAL FIX] 학습 효율을 위해 최종 DB 컬럼만 남김 (IMF 포함)

    # 1. DB에 남길 핵심 컬럼 정의 (IMF는 'ceemdan'에서 자동으로 병합된다고 가정)
    CORE_DB_COLUMNS = [
        # ID & Time (파이프라인 유지를 위해 필수)
        "battery_id", "cycle", 
        
        # Target & Core Physics (학습의 원재료)
        "capacity_ahr", "soh", "dcr", "lli", "lam", 
        
        # Environment & Key Trends (운용 기반)
        "ambient_temp_c", "voltage_measured_mean",
        "impedance_growth_log", "dcr_growth_log"
    ]
    
    # IMF/Residual 컬럼 추가 (ceemdan에서 가져왔다면)
    imf_cols = [col for col in m.columns if col.startswith('IMF') or col == 'Residual']
    
    # 최종 필터링 리스트 (순서 유지, 중복 제거)
    final_cols = list(pd.Series(CORE_DB_COLUMNS + imf_cols).drop_duplicates())
    
    # 실제로 존재하는 컬럼만 남김
    final_cols = [col for col in final_cols if col in m.columns]

    df_filtered = m[final_cols].copy()
    
    # [추가] 데이터 확인 (df.head() print)
    print("[Data Debug] Filtered DF Columns:", df_filtered.columns.tolist())
    print("[Data Debug] Filtered DF Head:\n", df_filtered.head())
    print("[Data Debug] Capacity Col: ", 'capacity_ahr' in df_filtered.columns or 'Capacity' in df_filtered.columns)

    return df_filtered

# --------------------------
# EOL / RUL Helpers
# --------------------------
def compute_eol_info(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out = {}
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
            
        max_cap = np.nanmax(cap[:10]) # 초기 용량은 앞부분에서
        eol_cap = 0.8 * max_cap

        # 기울기(Velocity) 계산 - v6 방식 단순화
        n = len(cap)
        tail = cap[int(n * 0.7):] # 후반부 기울기 사용
        cycles = np.arange(len(tail))
        
        if len(tail) < 2 or np.isnan(tail).all():
            vel = -1e-3
        else:
            tail_f = np.nan_to_num(tail, nan=float(np.nanmean(tail)))
            try:
                vel = linregress(cycles, tail_f).slope
            except ValueError:
                vel = -1e-3
            if abs(vel) < 1e-5: # 너무 작으면 보정
                vel = -1e-4

        out[str(bid)] = {"eol_cap": float(eol_cap), "vel": float(vel)}
    return out


def safe_compute_rul(capacity, eol_cap, vel):
    if np.isnan(capacity) or np.isnan(eol_cap) or np.isnan(vel):
        return 0.0
    # v6: (current - eol) / velocity
    # vel이 음수이므로 절대값 사용하거나 부호 주의
    return min(max(0.0, (capacity - eol_cap) / max(abs(vel), 1e-5)), 3000.0)

# --------------------------
# [v6 이식] Advanced Feature Extraction
# --------------------------
def compute_advanced_statistics(X: np.ndarray):
    """
    v6 코드의 'compute_advanced_statistics' 기능을 벡터화하여 이식.
    X shape: (Batch, Time, Features)
    Output: (Batch, Features * Num_Stats)
    """
    if X.size == 0:
        return np.zeros((X.shape[0], 1), dtype=np.float32)

    # 1. Basic Stats
    mean = np.nanmean(X, axis=1)
    std = np.nanstd(X, axis=1) + 1e-8
    mx = np.nanmax(X, axis=1)
    mn = np.nanmin(X, axis=1)
    
    # 2. Distribution Stats
    # scipy.stats 함수들은 axis 지원함
    skw = skew(np.nan_to_num(X, nan=0.0), axis=1, nan_policy='omit')
    krt = kurtosis(np.nan_to_num(X, nan=0.0), axis=1, nan_policy='omit')
    
    # 3. v6 Advanced Stats
    # Range
    rng = mx - mn
    
    # CV (Coefficient of Variation)
    cv = std / (np.abs(mean) + 1e-8)
    
    # Energy (Sum of squares)
    energy = np.nansum(X**2, axis=1)
    
    # Entropy (Approx) - 속도를 위해 Gaussian Entropy 근사 사용
    # Entropy ≈ 0.5 * log(2 * pi * e * std^2)
    entropy_approx = 0.5 * np.log(2 * np.pi * np.e * (std**2) + 1e-8)

    # 4. Linear Slope (Time trend)
    T = X.shape[1]
    t = np.arange(T, dtype=np.float32)
    t_mean = np.mean(t)
    t_var = np.var(t)
    
    # 각 피처별 평균 (y_mean)
    y_mean = np.nanmean(X, axis=1) # (Batch, Features)
    
    # 시간 t를 (1, T, 1) 모양으로 확장
    t_b = t.reshape(1, T, 1) # (1, Time, 1)
    
    # 분자: Sum[(t_i - t_mean) * (y_i - y_mean)]
    # (t_b - t_mean) shape: (1, T, 1)
    # (X - y_mean[:, None, :]) shape: (Batch, T, Features)
    numerator = np.nansum((t_b - t_mean) * (X - y_mean[:, None, :]), axis=1) # (Batch, Features)

    # 분모: Sum[(t_i - t_mean)^2] = T * var(t)
    denominator = T * t_var

    # 기울기 (Slope): (Batch, Features)
    # 분모가 0이거나 너무 작으면 0으로 처리하여 NaN/Inf 방지
    slope = numerator / (denominator + 1e-8)
    
    # 5. Concatenate
    stats_list = [mean, std, mx, mn, skw, krt, rng, cv, energy, entropy_approx, slope]
    
    # 최종 결과: (Batch, Total_Features)
    X_advanced = np.concatenate(stats_list, axis=1)

    return X_advanced # [수정] 벡터화된 결과 반환

def compute_physics_features(X_seq: np.ndarray, feature_cols: List[str]) -> pd.DataFrame:
    features = []

    # capacity index
    cap_idx = -1
    for name in ["capacity_ahr", "Capacity"]:
        if name in feature_cols:
            cap_idx = feature_cols.index(name)
            break

    # temperature index
    temp_idx = -1
    for name in ["ambient_temp_c", "ambient_temp"]:
        if name in feature_cols:
            temp_idx = feature_cols.index(name)
            break

    # voltage index (★ 새로 추가)
    vol_idx = -1
    for name in ["voltage_measured_mean", "voltage_max", "voltage_min"]:
        if name in feature_cols:
            vol_idx = feature_cols.index(name)
            break

    for i in range(X_seq.shape[0]):
        seq = X_seq[i]
        
        # 1. Capacity related
        if cap_idx != -1:
            cap = seq[:, cap_idx]
            cap_drop = cap[-1] - cap[0] 
            cap_drop_ratio = cap_drop / (cap[0] + 1e-8)
            cap_vel = (cap[-1] - cap[0]) / (seq.shape[0] + 1e-8) 
        else:
            cap_drop, cap_drop_ratio, cap_vel = 0.0, 0.0, 0.0
            
        # 2. Voltage derivative/Velocity (dVol/dTime) - V: 첫 번째 컬럼으로 가정
        # 2. Voltage related
        if vol_idx != -1:
            v = seq[:, vol_idx]
        else:
            v = np.zeros(seq.shape[0], dtype=np.float32)

        peaks, props = find_peaks(v, prominence=0.001)
        peak_count = len(peaks)
        peak_prom_sum = np.sum(props.get("prominences", [0.0]))
        
        # 3. Temperature related
        if temp_idx != -1:
            temp = seq[:, temp_idx]
            temp_max = np.max(temp)
            temp_range = np.ptp(temp) # Peak-to-Peak
        else:
            temp = np.zeros(seq.shape[0])
            temp_max = 0.0
            temp_range = 0.0

        # 특징 저장
        row = {
            'cap_drop': float(cap_drop),
            'cap_drop_ratio': float(cap_drop_ratio),
            'cap_vel': float(cap_vel),
            'v_peak_count': float(peak_count),
            'v_peak_prom': float(peak_prom_sum),
            'v_skew': float(skew(v, nan_policy='omit')),
            'v_kurt': float(kurtosis(v, nan_policy='omit')),
            'temp_max': float(temp_max),
            'temp_range': float(temp_range)
        }
        features.append(row)

    return pd.DataFrame(features).fillna(0.0).astype(np.float32)
# --------------------------
# Grouping Strategy
# --------------------------
# core/data.py 안의 group_data_by_battery 비슷한 함수

def group_data_by_battery_from_df(
    df: pd.DataFrame,
    cfg: Config,
    target_col: str,
    feature_cols: list,
    bid_col: str,
    cyc_col: str,
):
    """
    tabular_baseline_* 에서 load_db 로 가져온 df + target_col + feature_cols 를
    메타 학습용 sequence 그룹으로 바꿔주는 함수.
    """
    # seq 브랜치 / dnn 브랜치 feature 분리
    # 일단은 심플하게:
    base_tf_cols = feature_cols               # 시퀀스에 들어갈 컬럼들
    dnn_source_cols = feature_cols           # 요약統계 낼 컬럼들

    imf_cols = [c for c in df.columns if c.startswith("IMF")]
    tf_cols = sorted(list(set(base_tf_cols + imf_cols)))   # seq branch (물리 + IMF)

    grouped = {}

    for bid, g in df.groupby(bid_col):
        bid = str(bid)
        g = g.sort_values(cyc_col)

        if len(g) < cfg.seq_len + cfg.k_shot + cfg.q_query:
            continue

        # RUL 시퀀스: CSV에 있는 것 그대로 사용
        r_seq = g[target_col].values.astype(np.float32)

        # 입력 시퀀스
        s_np = g[tf_cols].values.astype(np.float32)              # seq branch
        d_np = g[dnn_source_cols].values.astype(np.float32)      # dnn branch(원자료)

        Xs, Xd, yr = [], [], []
        for i in range(len(g) - cfg.seq_len):
            Xs.append(s_np[i:i + cfg.seq_len])
            Xd.append(d_np[i:i + cfg.seq_len])
            yr.append(r_seq[i + cfg.seq_len - 1])  # window 끝 시점의 RUL

        if len(Xs) < cfg.k_shot + cfg.q_query:
            continue

        Xs_arr = np.array(Xs, dtype=np.float32)   # (N_seq, T, F_seq)
        Xd_arr = np.array(Xd, dtype=np.float32)   # (N_seq, T, F_dnn)

        # advanced stats + physics feature 그대로 사용
        stats_feat   = compute_advanced_statistics(Xd_arr)       # (N_seq, D_stats)
        physics_feat = compute_physics_features(Xs_arr, tf_cols).values  # (N_seq, D_phys)

        X_sum_arr = np.hstack([stats_feat, physics_feat])        # (N_seq, D_sum)

        grouped[bid] = {
            "seq": Xs_arr,
            "sum": X_sum_arr,
            "rul": np.array(yr, dtype=np.float32),
            "max_rul": float(r_seq.max() + 1e-8),
            "capacity_curve": g["capacity_ahr"].values.astype(np.float32)
                              if "capacity_ahr" in g.columns else None,
            "cycle": g[cyc_col].values.astype(np.float32),
        }

    return grouped

# --------------------------
# Meta Dataset (기존 유지)
# --------------------------
class MetaBatteryDataset(Dataset):
    def __init__(self, grouped_data, battery_ids, c: Config, seq_scaler, sum_scaler,
                 mode='train', max_rul_train=None):
        self.gr = grouped_data
        self.bids = battery_ids
        self.c = c
        self.mode = mode
        self.seq_scaler = seq_scaler
        self.sum_scaler = sum_scaler
        self.max_rul_train = max_rul_train

    def __len__(self):
        return len(self.bids)

    def _select_support_indices(self, support_prefix_len):
        k = self.c.k_shot
        if support_prefix_len <= 0:
            return np.array([], dtype=int)
        if support_prefix_len <= k:
            return np.arange(support_prefix_len)
        
        if self.c.support_strategy == "mixed":
            # [추가] Mix: 50% recent + 50% random
            half_k = k // 2
            recent_idx = np.arange(support_prefix_len - half_k, support_prefix_len)
            random_idx = np.sort(np.random.choice(support_prefix_len, size=k - half_k, replace=False))
            mixed_idx = np.unique(np.concatenate([recent_idx, random_idx]))
            return mixed_idx[:k]  # k개로 자르기
        
        if self.c.support_strategy == "recent":
            return np.arange(support_prefix_len - k, support_prefix_len)
        if self.c.support_strategy == "random":
            return np.sort(np.random.choice(support_prefix_len, size=k, replace=False))
        return np.linspace(0, support_prefix_len - 1, k, dtype=int)

    def __getitem__(self, idx):
        bid = self.bids[idx]
        data = self.gr[bid]
        n = len(data["seq"])

        r = np.random.uniform(self.c.prefix_min_ratio, self.c.prefix_max_ratio) \
            if self.mode == "train" else self.c.val_prefix_ratio

        prefix_len = int(n * r)
        min_need = self.c.k_shot + self.c.q_query + 1
        prefix_len = min(n, max(prefix_len, min_need))

        q_query = self.c.q_query
        support_prefix_len = prefix_len - q_query
        if support_prefix_len <= 0:
            support_prefix_len = max(prefix_len // 2, 1)
            q_query = prefix_len - support_prefix_len

        s_idx = self._select_support_indices(support_prefix_len)
        q_idx = np.arange(support_prefix_len, prefix_len)

        if len(s_idx) < 1 or len(q_idx) < 1:
            # Fallback for empty
            feat_dim = data['seq'].shape[-1]
            sum_dim = data['sum'].shape[-1]
            return {
                's_seq': torch.zeros((1, self.c.seq_len, feat_dim)),
                's_sum': torch.zeros((1, sum_dim)),
                's_rul': torch.zeros(1),
                'q_seq': torch.zeros((1, self.c.seq_len, feat_dim)),
                'q_sum': torch.zeros((1, sum_dim)),
                'q_rul': torch.zeros(1),
                'max_rul': 1.0,
                'cycles': None,
                'effective': False,
            }

        s_seq = self.seq_scaler.transform(data['seq'][s_idx])
        s_sum = self.sum_scaler.transform(data['sum'][s_idx])
        s_rul = scale_rul_array(data['rul'][s_idx], self.c.rul_mode, self.max_rul_train)

        q_seq = self.seq_scaler.transform(data['seq'][q_idx])
        q_sum = self.sum_scaler.transform(data['sum'][q_idx])
        q_rul = scale_rul_array(data['rul'][q_idx], self.c.rul_mode, self.max_rul_train)

        # [추가] cycles 추출 (support + query 결합)
        cycles = np.concatenate([data['cycle'][s_idx], data['cycle'][q_idx]]) if 'cycle' in data else None

        return {
            's_seq': torch.from_numpy(s_seq).float(),
            's_sum': torch.from_numpy(s_sum).float(),
            's_rul': torch.from_numpy(s_rul).float(),
            'q_seq': torch.from_numpy(q_seq).float(),
            'q_sum': torch.from_numpy(q_sum).float(),
            'q_rul': torch.from_numpy(q_rul).float(),
            'max_rul': float(data['max_rul']),
            'cycles': torch.from_numpy(cycles).float() if cycles is not None else None,  # [추가] cycles Torch로 변환
            'effective': True,
        }