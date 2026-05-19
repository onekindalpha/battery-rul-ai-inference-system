# deep_learning/core/meta_db_loader.py
"""통합 NASA + CACLE cycle-level DB 로더.

- (preprocessing)tabular_baseline_rul.py 안의 ``load_db`` / ``select_feature_columns`` 를
  meta-learning 파이프라인에서 재사용하기 위한 헬퍼 모듈.
- ``Config.dataset_source`` 에 따라
  - "nasa"  : NASA 만 사용
  - "cacle" : CACLE 만 사용
  - "both"  : NASA + CACLE 을 concat 해서 함께 사용
  을 지원한다.
"""

from typing import List, Tuple
import pandas as pd

from deep_learning.core.config import Config

# deep_learning/db/preprocessing/tabular_baseline_rul.py 안의 공용 헬퍼
from deep_learning.db.preprocessing.tabular_baseline_rul import (
    load_db,
    select_feature_columns,
)
def _load_single_source(dataset: str, cfg: Config) -> Tuple[pd.DataFrame, str, List[str]]:
    df, target_col = load_db(
        dataset=dataset,
        db_type=cfg.db_type,
        use_ceemdan=cfg.use_ceemdan,
    )

    # ---------------------------------------------------------
    # [수정] 치명적 오류 데이터 필터링 로직
    #  - 하드코딩 NASA Outlier: B0047
    #  - + 자동 탐지: "초기 RUL <= 0" 인 배터리 전부 제거
    # ---------------------------------------------------------
    bid_col = "battery_id" if "battery_id" in df.columns else "battery"

    # ✅ cycle 인덱스 컬럼 robust하게 찾기
    if "cycle" in df.columns:
        cyc_col = "cycle"
    elif "cycle_num" in df.columns:
        cyc_col = "cycle_num"
    elif "cycle_index" in df.columns:
        cyc_col = "cycle_index"
    else:
        cyc_col = None  # 최악의 경우 대비

    # NASA 데이터셋의 알려진 이상치 리스트 (수동 등록)
    bad_batteries = ["B0047"]

    if dataset.lower() == "nasa":
        # 1) 자동 이상치 탐지: 각 배터리의 "첫 사이클"에서 RUL(target_col) <= 0 인 배터리
        if target_col in df.columns:
            if cyc_col is not None:
                first_rows = (
                    df.sort_values(cyc_col)
                      .groupby(bid_col, as_index=False)
                      .first()
                )
            else:
                # cycle 컬럼 못 찾으면, 그냥 현재 순서 기준으로 첫 row 사용
                first_rows = (
                    df.groupby(bid_col, as_index=False)
                      .first()
                )

            auto_bad = (
                first_rows[first_rows[target_col] <= 0][bid_col]
                .astype(str)
                .tolist()
            )

            extra_bad = [b for b in auto_bad if b not in bad_batteries]
            if extra_bad:
                print(
                    f"[META-LOAD] Auto-detected batteries with initial {target_col} <= 0: "
                    f"{extra_bad}"
                )
                bad_batteries += extra_bad

        before_len = len(df)
        df = df[~df[bid_col].isin(bad_batteries)].copy()
        after_len = len(df)
        if before_len != after_len:
            print(
                f"[META-LOAD] Filtered out bad batteries {bad_batteries}: "
                f"rows {before_len} -> {after_len}"
            )

    feature_cols: List[str] = select_feature_columns(
        df=df,
        db_type=cfg.db_type,
        feature_set=cfg.feature_set,
    )

    return df, target_col, feature_cols


def load_cycle_db_for_meta(cfg: Config):
    """Config 기반 cycle-level DB 로드 유틸.

    Returns
    -------
    df : pandas.DataFrame
        cycle-level 데이터프레임 (NASA, CACLE 혹은 둘 다)
    target_col : str
        RUL(target) 컬럼 이름
    feature_cols : List[str]
        모델 입력에 사용할 feature 컬럼 리스트
    bid_col : str
        배터리 ID 컬럼 이름
    cyc_col : str
        cycle index 컬럼 이름
    """

    if cfg.dataset_source in ("nasa", "cacle"):
        # 기존 단일 데이터셋 로직
        df, target_col, feature_cols = _load_single_source(cfg.dataset_source, cfg)

    elif cfg.dataset_source == "both":
        # NASA + CACLE 각각 로드 후 concat
        df_nasa, target_n, feat_n = _load_single_source("nasa", cfg)
        df_cacle, target_c, feat_c = _load_single_source("cacle", cfg)

        if target_n != target_c:
            raise ValueError(
                f"target_col mismatch between NASA({target_n}) and CACLE({target_c})"
            )
        target_col = target_n

        # 두 feature list 의 교집합을 사용 (순서는 NASA 기준)
        common_feats: List[str] = [c for c in feat_n if c in feat_c]
        if not common_feats:
            raise ValueError(
                "No common feature columns between NASA and CACLE. "
                "Check select_feature_columns / feature_set configuration."
            )

        # concat + 공통 feature 사용
        df = pd.concat([df_nasa, df_cacle], ignore_index=True)

        # 혹시 누락된 컬럼이 있으면 자동으로 걸러지도록
        feature_cols = [c for c in common_feats if c in df.columns]

        print(
            f"[META-LOAD] combined NASA+CACLE: "
            f"N_nasa={len(df_nasa)}, N_cacle={len(df_cacle)}, N_total={len(df)}"
        )
    else:
        raise ValueError(f"Unknown dataset_source: {cfg.dataset_source}")

    # 배터리 ID / cycle 컬럼 이름 자동 감지 (기존 구현 유지) :contentReference[oaicite:1]{index=1}
    bid_col = "battery_id" if "battery_id" in df.columns else "battery"
    cyc_col = "cycle" if "cycle" in df.columns else "cycle_num"

    print("[META-LOAD] dataset_source:", cfg.dataset_source)
    print("[META-LOAD] target_col   :", target_col)
    print("[META-LOAD] num features :", len(feature_cols))
    print("[META-LOAD] feature_cols :", feature_cols)

    return df, target_col, feature_cols, bid_col, cyc_col
