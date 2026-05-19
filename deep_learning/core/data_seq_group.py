# deep_learning/core/data_seq_group.py

from typing import Dict, List

import numpy as np
import pandas as pd

from deep_learning.core.config import Config
from deep_learning.core.data import (
    compute_advanced_statistics,
    compute_physics_features,
)

def group_data_by_battery_from_df(
    df: pd.DataFrame,
    cfg: Config,
    target_col: str,
    feature_cols: List[str],
    bid_col: str,
    cyc_col: str,
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    cycle-level dataframe(df) → 배터리별 sequence 그룹으로 변환.

    반환되는 grouped[battery_id] 딕셔너리는:
      - "seq": (N_seq, T, F_seq)  : transformer branch 입력
      - "sum": (N_seq, D_sum)     : DNN branch summary feature
      - "rul": (N_seq,)           : window 끝 시점의 RUL (scale 전)
      - "max_rul": float          : 이 배터리의 max RUL
      - "capacity_curve": (N_cycle,) or None
      - "cycle": (N_cycle,)       : 원본 cycle index

    + drop_zero_tail 옵션:
      - cfg.drop_zero_tail == True 이면,
        한 배터리 내에서 RUL=0인 윈도우가 tail에서 여러 개 나올 때
        첫 0 RUL 윈도우 하나만 남기고 나머지 0-RUL 윈도우는 버린다.
    """
    # seq branch / summary branch에 넣을 base feature들
    base_tf_cols = list(feature_cols)
    dnn_source_cols = list(feature_cols)

    # IMF 계열은 seq branch에만 추가
    imf_cols = [c for c in df.columns if c.startswith("IMF")]
    tf_cols = sorted(set(base_tf_cols + imf_cols))  # seq branch feature들

    grouped = {}
    drop_zero_tail = bool(getattr(cfg, "drop_zero_tail", False))

    for bid, g in df.groupby(bid_col):
        
        r_seq = g[target_col].values.astype(np.float32)

        # [치명적 오류 수정] Max RUL이 0이거나 너무 작으면 이 배터리는 학습 불가
        current_max_rul = float(r_seq.max())
        if current_max_rul < 1.0:
            print(f"[WARN] Battery {bid} excluded: max_rul is too low ({current_max_rul})")
            continue

        # RUL 시퀀스 (scale 전 원본)
        r_seq = g[target_col].values.astype(np.float32)

        # 입력 시퀀스 (seq branch / dnn branch)
        s_np = g[tf_cols].values.astype(np.float32)          # (N_cycle, F_seq)
        d_np = g[dnn_source_cols].values.astype(np.float32)  # (N_cycle, F_dnn)

        Xs, Xd, yr = [], [], []

        last_zero = False  # tail에서 RUL=0이 연속되는지 체크용

        # 기존 코드와 동일하게: window 끝 인덱스 = i + seq_len - 1
        # window 개수: len(g) - seq_len
        for i in range(len(g) - cfg.seq_len):
            idx_y = i + cfg.seq_len - 1
            y_val = float(r_seq[idx_y])

            if drop_zero_tail and y_val <= 0.0:
                # 이미 바로 직전에도 0이었다 → 이 윈도우는 tail 중복 0RUL → 스킵
                if last_zero:
                    continue
                # 첫 번째 0RUL 윈도우는 남기고 플래그만 세팅
                last_zero = True
            else:
                last_zero = False

            Xs.append(s_np[i : i + cfg.seq_len])
            Xd.append(d_np[i : i + cfg.seq_len])
            yr.append(y_val)

        # support(k_shot) + query(q_query) 합보다 적으면 이 배터리는 사용 불가
        if len(Xs) < cfg.k_shot + cfg.q_query:
            continue

        Xs_arr = np.asarray(Xs, dtype=np.float32)
        Xd_arr = np.asarray(Xd, dtype=np.float32)
        # summary branch: advanced stats + physics feature
        stats_feat = compute_advanced_statistics(Xd_arr)  # (N_seq, D_stats)
        physics_feat_df = compute_physics_features(Xs_arr, tf_cols)
        physics_feat = physics_feat_df.values              # (N_seq, D_phys)

        X_sum_arr = np.hstack([stats_feat, physics_feat])  # (N_seq, D_sum)

        grouped[bid] = {
            "seq": Xs_arr,
            "sum": X_sum_arr,
            "rul": np.asarray(yr, dtype=np.float32),
            "max_rul": current_max_rul + 1e-8, # 안전장치 유지
            "capacity_curve": g["capacity_ahr"].values.astype(np.float32) if "capacity_ahr" in g.columns else None,
            "cycle": g[cyc_col].values.astype(np.float32),
        }

    print(f"[META-GROUP] built {len(grouped)} batteries")
    return grouped