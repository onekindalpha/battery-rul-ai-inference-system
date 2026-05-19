import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
import glob
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Optional: non-blocking auto-refresh for auto-play
try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
except Exception:
    st_autorefresh = None

import glob
from pathlib import Path
# deep_learning 패키지 의존성(torch 등) 없이도 precomputed 모드를 돌릴 수 있게,
# Precomputed loader는 optional import로 처리합니다.
try:
    from deep_learning.core.rul_precomputed_loader_restored import PrecomputedRULLoader as _PrecomputedRULLoader  # type: ignore
except Exception as _e_pre:
    _PrecomputedRULLoader = None
    _PRECOMPUTED_IMPORT_ERROR = _e_pre

if _PrecomputedRULLoader is not None:
    PrecomputedRULLoader = _PrecomputedRULLoader
else:
    class PrecomputedRULLoader:  # fallback
        def __init__(self, export_root: Path):
            self.export_root = Path(export_root)

        def _candidate_paths(self, bid: str, r_ratio: float):
            bid = str(bid)
            # new format: B0018_r0.25.json
            yield self.export_root / f"{bid}_r{float(r_ratio):.2f}.json"
            # old format: battery_B0018_r0p25.json
            yield self.export_root / f"battery_{bid}_r{str(float(r_ratio)).replace('.', 'p')}.json"
            # old index format: battery_B0018_r0p25 (maybe different)
            # fallback glob
            tag1 = f"r{float(r_ratio):.2f}"
            tag2 = f"r{str(float(r_ratio)).replace('.', 'p')}"
            for p in glob.glob(str(self.export_root / f"*{bid}*{tag1}*.json")):
                yield Path(p)
            for p in glob.glob(str(self.export_root / f"*{bid}*{tag2}*.json")):
                yield Path(p)

        def has_precomputed(self, bid: str, r_ratio: float) -> bool:
            for p in self._candidate_paths(bid, r_ratio):
                if p.exists():
                    return True
            return False

        def load(self, bid: str, r_ratio: float):
            for p in self._candidate_paths(bid, r_ratio):
                if p.exists():
                    with open(p, "r") as f:
                        return json.load(f)
            return None



# precomputed json index (battery → {r_ratio: filepath})
def load_precomputed_index(folder: Path):
    out = {}
    files = glob.glob(str(folder / "battery_*_r*.json"))
    for f in files:
        stem = Path(f).stem  # battery_B0018_r0p15
        _, bid, rtag = stem.split("_")
        r = float(rtag.replace("r", "").replace("p", "."))
        if bid not in out:
            out[bid] = {}
        out[bid][r] = f
    return out

# -------------------------------------------------
# Paths / sys.path 설정
# -------------------------------------------------
FILE_DIR = Path(__file__).resolve().parent  # .../v11/deep_learning/core
PROJECT_ROOT = FILE_DIR.parent.parent       # .../v11

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ASSETS_DIR = FILE_DIR / "assets"
LOADING_GIF = ASSETS_DIR / "loading.gif"
HEALTH_HIGH_GIF = ASSETS_DIR / "high.gif"
HEALTH_MED_GIF = ASSETS_DIR / "medium.gif"  # EV6 이미지는 예시용으로만 사용
HEALTH_LOW_GIF = ASSETS_DIR / "low.gif"

BMAML_DIR = FILE_DIR / "dashboard_export" / "bmaml"
CKPT_DEFAULT = FILE_DIR / "core_checkpoints" / "nasa_bmaml_best_re.pt"
SHAP_JSON = FILE_DIR / "shap_outputs" / "bmaml_shap_seq_feature_importance.json"

FEATURE_STATS_JSON_PATH = FILE_DIR / "analysis" / "feature_rul_stats.json"
FEATURE_STATS_CSV_PATH = FILE_DIR / "analysis" / "feature_rul_stats.csv"

# Prefer CSV if available (easier to inspect/version)
FEATURE_STATS_PATH = FEATURE_STATS_CSV_PATH if FEATURE_STATS_CSV_PATH.exists() else FEATURE_STATS_JSON_PATH
NASA_FEATURES_PATH = FILE_DIR / "analysis" / "nasa_features_rul.csv"

EXPORT_TEST_BATTERIES = {"B0018", "B0033", "B0043", "B0055"}  # 첫 화면용

# (legacy) preferred list was hard-coded; keep initial demo batteries only
preferred = sorted(list(EXPORT_TEST_BATTERIES))

DEFAULT_R_RATIO = 0.25
DEFAULT_CYCLES_PER_DAY = 1.0  # 기본 하루 평균 1 사이클

# runtime few-shot re-adaptation (torch 필요). 없으면 precomputed 모드만 사용.
try:
    from deep_learning.core.prefix_inference_viz_meta_restored_v3_pyc import (  # type: ignore
        build_model_and_grouped,
        make_task_prefix,
        run_adapt_and_predict,
    )
    RUNTIME_AVAILABLE = True
    _RUNTIME_IMPORT_ERROR = None
except Exception as _e_rt:
    build_model_and_grouped = None
    make_task_prefix = None
    run_adapt_and_predict = None
    RUNTIME_AVAILABLE = False
    _RUNTIME_IMPORT_ERROR = _e_rt

AUTO_PLAY_DELAY_SEC = 0.6   # 자동 재생 프레임 간 간격 (초)
AUTO_PLAY_STEP = 1          # 한 번에 몇 사이클씩 움직일지

# -------------------------------------------------
# 시나리오 빌더에 사용할 feature 정의
# (feature_rul_stats.json + causal 피처 기준)
# -------------------------------------------------
SCENARIO_FEATURES = {
    "soh": {
        "label": "배터리 건강도 SoH (0~1)",
        "fallback_min": 0.6,
        "fallback_max": 1.0,
        "step": 0.01,
    },
    "capacity_mean": {
        "label": "평균 용량 (Ah)",
        "fallback_min": 1.0,
        "fallback_max": 3.5,
        "step": 0.05,
    },
    "voltage_min": {
        "label": "최소 전압 (V)",
        "fallback_min": 2.0,
        "fallback_max": 3.5,
        "step": 0.02,
    },
    "current_mean": {
        "label": "평균 전류 (A)",
        "fallback_min": -6.0,
        "fallback_max": 0.0,
        "step": 0.1,
    },
    "current_min": {
        "label": "최대 방전 전류 (A, 가장 음수)",
        "fallback_min": -10.0,
        "fallback_max": 0.0,
        "step": 0.2,
    },
    "dcr": {
        "label": "직류 저항 DCR (Ω)",
        "fallback_min": 0.0,
        "fallback_max": 1.0,
        "step": 0.001,
    },
        # 그 다음 인피던스 합
    "impedance_sum": {
        "label": "임피던스 합 (Re + Rct, Ω)",
        "fallback_min": 0.0,
        "fallback_max": 2.0,
        "step": 0.001,
    },
        # regen_strength → 정의상 "최대 용량 대비 드롭" 이라 이름만 바꿔서 사용
    "regen_strength": {
        "label": "누적 용량 손실 (최고 용량 대비)",
        "fallback_min": 0.0,
        "fallback_max": 1.5,
        "step": 0.05,
    },

    # 🔽 여기서부터 순서 변경: 먼저 주변 온도
    "ambient_temp_c": {
        "label": "주변 온도 (°C)",
        "fallback_min": 10.0,
        "fallback_max": 45.0,
        "step": 1.0,
    },
    "temp_rise_cycle": {
        "label": "사이클 온도 상승 (셀 - 주변, °C)",
        "fallback_min": 0.0,
        "fallback_max": 20.0,
        "step": 0.5,
    },
}


class ScenarioContext:
    """
    - (battery_id, cycle) 컨텍스트가 바뀌면
      * 모든 Scenario 값을 Baseline으로 리셋
      * 각 슬라이더의 session_state도 Baseline 값으로 맞춰줌
    - 같은 컨텍스트에서는 사용자가 바꾼 Scenario 값을 그대로 유지
    """

    STATE_CTX = "scenario_ctx"
    STATE_VALUES = "scenario_values"

    def __init__(self, battery_id: str, cycle: int):
        self.ctx = (str(battery_id), int(cycle))
        prev = st.session_state.get(self.STATE_CTX)
        self.ctx_changed = (prev != self.ctx)

        if self.ctx_changed:
            # 컨텍스트가 바뀌면, 이전 Scenario 값은 버리고 다시 시작
            st.session_state[self.STATE_CTX] = self.ctx
            st.session_state[self.STATE_VALUES] = {}
        elif self.STATE_VALUES not in st.session_state:
            st.session_state[self.STATE_VALUES] = {}

    @property
    def values(self) -> Dict[str, float]:
        return st.session_state[self.STATE_VALUES]

    def register_baseline(self, feat_key: str, baseline_value: float) -> float:
        """
        각 피처마다 Baseline 값 등록 + Scenario 초기값 결정.

        - 컨텍스트가 바뀐 경우: Scenario = Baseline 으로 강제 세팅
        - 컨텍스트 그대로인데 처음 보는 피처: Scenario = Baseline
        - 이미 Scenario 값이 있는 피처: 그대로 유지
        """
        baseline_value = float(baseline_value)
        vals = self.values

        if self.ctx_changed:
            # 새 배터리/사이클이면 슬라이더도 Baseline으로 강제 이동
            st.session_state[f"{feat_key}_scenario"] = baseline_value
            vals[feat_key] = baseline_value
            return baseline_value

        # 같은 컨텍스트인데 아직 값이 없으면 Baseline으로 초기화
        if feat_key not in vals:
            vals[feat_key] = baseline_value
        return float(vals[feat_key])

    def update_from_slider(self, feat_key: str, slider_value: float):
        """슬라이더에서 읽은 값을 Scenario 값으로 반영."""
        self.values[feat_key] = float(slider_value)

    def reset_to_baseline(self, baseline_dict: Dict[str, float]):
        """
        Scenario 리셋 버튼에서 사용:
        - 모든 피처의 Scenario 값을 현재 Baseline으로 되돌리고
        - 슬라이더 position도 Baseline으로 맞춤
        """
        new_vals = {}
        for feat_key, base in baseline_dict.items():
            v = float(base)
            new_vals[feat_key] = v
            st.session_state[f"{feat_key}_scenario"] = v
        st.session_state[self.STATE_VALUES] = new_vals

# 시나리오에서 환경/열화 상태로만 보는 피처(사용자 조작 잠금)
# 시나리오에서 잠글 피처 (온도 2개만 고정, 누적 용량 손실은 사용자가 조절)
LOCKED_SCENARIO_FEATURES = {"ambient_temp_c", "temp_rise_cycle"}


class ScenarioManager:
    """
    - 배터리/사이클 컨텍스트(bid, cycle)가 바뀌면
      → 시나리오 슬라이더 값을 전부 초기화.
    - 현재 컨텍스트에서 사용자가 슬라이더를 건드린 피처만
      → '바뀐 피처와 RUL 영향' 및 ΔRUL 계산에 반영.
    """

    def __init__(
        self,
        feature_keys,
        locked_features=None,
        ctx_state_key="scenario_ctx_key",
        flags_state_key="scenario_changed_flags",
        eps: float = 1e-6,
    ):
        self.feature_keys = list(feature_keys)
        self.locked_features = set(locked_features or [])
        self.ctx_state_key = ctx_state_key
        self.flags_state_key = flags_state_key
        self.eps = eps

    def _slider_key(self, feat_key: str) -> str:
        return f"{feat_key}_scenario"

    def set_context(self, battery_id: str, cycle: int) -> bool:
        """
        현재 컨텍스트(bid, cycle)를 세션에 저장.
        이전과 다르면:
        - 모든 시나리오 슬라이더 상태 삭제
        - 변경 플래그 초기화
        """
        ctx_id = f"{battery_id}:{cycle}"
        prev = st.session_state.get(self.ctx_state_key)
        changed = prev != ctx_id

        if changed:
            st.session_state[self.ctx_state_key] = ctx_id

            # 슬라이더 값 초기화
            for fk in self.feature_keys:
                sk = self._slider_key(fk)
                if sk in st.session_state:
                    del st.session_state[sk]

            # 변경 플래그 초기화
            st.session_state[self.flags_state_key] = {
                fk: False for fk in self.feature_keys
            }
        else:
            # 처음 진입했을 수도 있으니 한번은 만들어 둠
            if self.flags_state_key not in st.session_state:
                st.session_state[self.flags_state_key] = {
                    fk: False for fk in self.feature_keys
                }

        return changed

    def update_changed_flags(
        self,
        baseline_vals: Dict[str, float],
        scenario_vals: Dict[str, float],
    ) -> None:
        """
        baseline vs scenario 를 비교해서
        어떤 피처가 실제로 바뀌었는지 플래그 갱신.
        """
        flags = st.session_state.get(
            self.flags_state_key, {fk: False for fk in self.feature_keys}
        )

        for fk in self.feature_keys:
            b = float(baseline_vals.get(fk, 0.0))
            s = float(scenario_vals.get(fk, 0.0))
            flags[fk] = abs(s - b) >= self.eps

        st.session_state[self.flags_state_key] = flags

    def get_changed_flags(self) -> Dict[str, bool]:
        return st.session_state.get(
            self.flags_state_key, {fk: False for fk in self.feature_keys}
        )

    def reset_current(self, baseline_vals: Dict[str, float]) -> None:
        """
        현재 배터리/사이클 컨텍스트에서
        - 모든 슬라이더를 Baseline 값으로 되돌림
        - 변경 플래그도 전부 False
        (UI 틀은 그대로 유지)
        """
        for fk in self.feature_keys:
            sk = self._slider_key(fk)
            if fk in baseline_vals:
                st.session_state[sk] = float(baseline_vals[fk])

        st.session_state[self.flags_state_key] = {
            fk: False for fk in self.feature_keys
        }

# 피처별 기본 가이드 (텍스트 고정, 시나리오 버튼 눌러도 유지)
FEATURE_GUIDES = {
    "soh": (
        "SoH를 가능한 크게 유지해 보세요. (1에 가까울수록 수명이 긴 편입니다.)"
    ),
    "capacity_mean": (
        "평균 용량이 클수록 유리합니다. 값이 너무 작지 않도록 유지해 보세요."
    ),
    "voltage_min": (
        "최소 전압을 너무 낮추지 말고, 약간 높게 잡아서 깊은 방전을 피하세요."
    ),
    "current_mean": (
        "평균 전류의 절대값을 줄이는 쪽이 유리합니다. 너무 세게 방전하지 않도록 해 보세요."
    ),
    "current_min": (
        "피크 방전 전류(가장 음수 값)를 줄이세요. 순간적으로 너무 큰 전류가 흐르지 않게 하는 쪽이 좋습니다."
    ),
    "dcr": (
        "DCR 값이 낮을수록 좋습니다. DCR이 크게 증가한 상태에서 오래 쓰는 시나리오는 피하세요."
    ),
    "impedance_sum": (
        "임피던스 합이 낮을수록 좋습니다. 값이 많이 커진 구간에 오래 머무르지 않도록 가정해 보세요."
    ),
    "ambient_temp_c": (
        "너무 뜨거운 환경은 피하고, 대략 20~30°C 정도의 중간 온도를 목표로 잡아 보세요."
    ),
    "temp_rise_cycle": (
        "사이클 동안 온도 상승이 작을수록 유리합니다. 냉각을 잘 해서 온도 차이를 줄이는 방향으로 가정하세요."
    ),
    "regen_strength": (
        "누적 용량 손실 값이 작을수록 좋습니다. 이 값을 낮추는 쪽으로 시나리오를 잡아 보세요."
    ),
}


# -------------------------------------------------
# Small CSS
# -------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 0.6rem;
        padding-bottom: 1.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------
# Data loaders
# -------------------------------------------------

def load_battery_records(folder: Path) -> Dict[str, dict]:
    """기존 export 스크립트가 만들어둔 JSON들 로드 (빠름)."""
    if not folder.exists():
        return {}
    pattern = str(folder / "battery_*.json")
    paths = sorted(glob.glob(pattern))
    records: Dict[str, dict] = {}
    for p in paths:
        with open(p, "r") as f:
            data = json.load(f)
            bid = str(data["battery_id"])
            records[bid] = data
    return records


def filter_export_demo(records: Dict[str, dict]) -> Dict[str, dict]:
    subset = {bid: rec for bid, rec in records.items() if bid in EXPORT_TEST_BATTERIES}
    return subset or records


def load_shap_importance(path: Path) -> Tuple[List[str], np.ndarray]:
    if not path.exists():
        return [], np.array([])
    with open(path, "r") as f:
        data = json.load(f)
    names = data.get("feature_names", [])
    vals = np.asarray(data.get("importance", []), dtype=float)
    if len(names) != len(vals):
        n = min(len(names), len(vals))
        names = names[:n]
        vals = vals[:n]
    return names, vals


@st.cache_resource
def load_cycle_features():
    """nasa_features_rul.csv: 배터리별·사이클별 피처 테이블"""
    if not NASA_FEATURES_PATH.exists():
        return None
    df = pd.read_csv(NASA_FEATURES_PATH)
    # 컬럼 통일 (battery_id, cycle)
    df["battery_id"] = df["battery"].astype(str)
    df.rename(columns={"cycle_num": "cycle"}, inplace=True)
    return df


class CycleFeatureContext:
    """현재 배터리/사이클 메타 피처 + 파이프라인 SoH 래핑."""

    def __init__(self, battery_id: str, cycle: int):
        self.battery_id = str(battery_id)
        self.cycle = int(cycle)
        self.row = None

        df = load_cycle_features()
        if df is not None:
            mask = (df["battery_id"] == self.battery_id) & (df["cycle"] == self.cycle)
            sub = df.loc[mask]
            if not sub.empty:
                self.row = sub.iloc[0]

    @property
    def soh(self) -> float:
        """파이프라인에서 계산된 soh (없으면 NaN)."""
        if self.row is None or "soh" not in self.row:
            return float("nan")
        try:
            return float(self.row["soh"])
        except Exception:
            return float("nan")


@st.cache_resource
def load_feature_stats(path: Path):
    """feature_rul_stats (json/csv) 로드.

    - json: {feature: {slope_rul_per_unit: ... , ...}}
    - csv : columns include: feature, slope_rul_per_unit, ...
    """
    if not path.exists():
        return None
    if path.suffix.lower() == ".json":
        with open(path, "r") as f:
            return json.load(f)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        out = {}
        if "feature" not in df.columns:
            return None
        for _, r in df.iterrows():
            feat = str(r["feature"])
            out[feat] = {k: (float(r[k]) if k in r and pd.notna(r[k]) else r.get(k)) for k in df.columns if k != "feature"}
        return out
    # unknown format
    return None


FEATURE_STATS = load_feature_stats(FEATURE_STATS_PATH)
shap_names, shap_vals = load_shap_importance(SHAP_JSON)

@st.cache_resource
def compute_stage_slopes():
    """수명 단계(early/mid/late)별 feature→RUL 선형 민감도(기울기) 계산.

    목적: 전역 평균 slope(데이터셋 편향)보다, 현재 life-stage에 더 근접한 local/global 중간 해석을 제공.
    """
    df = load_cycle_features()
    if df is None or "rul_cycles" not in df.columns:
        return {}

    # battery별 관측 수명(최대 cycle)
    max_cycle = df.groupby("battery_id")["cycle"].max().rename("cycle_max")
    df2 = df.join(max_cycle, on="battery_id")
    df2 = df2[df2["cycle_max"].notna()]
    df2["life_frac"] = df2["cycle"].astype(float) / df2["cycle_max"].astype(float).replace(0.0, np.nan)

    def stage_mask(stage: str):
        if stage == "early":
            return df2["life_frac"] <= 0.30
        if stage == "mid":
            return (df2["life_frac"] > 0.30) & (df2["life_frac"] <= 0.70)
        return df2["life_frac"] > 0.70  # late

    stages = ["early", "mid", "late"]
    slopes = {s: {} for s in stages}

    # helper: simple least-squares slope (y ~ a*x + b)
    def safe_slope(x: np.ndarray, y: np.ndarray) -> float:
        if x.size < 50:
            return float("nan")
        vx = np.var(x)
        if not np.isfinite(vx) or vx < 1e-12:
            return float("nan")
        cov = np.cov(x, y, ddof=0)[0, 1]
        return float(cov / vx)

    for stage in stages:
        sub = df2.loc[stage_mask(stage)]
        for feat_key in SCENARIO_FEATURES.keys():
            if feat_key not in sub.columns:
                continue
            x = sub[feat_key].astype(float).values
            y = sub["rul_cycles"].astype(float).values
            mask = np.isfinite(x) & np.isfinite(y)
            x, y = x[mask], y[mask]
            s = safe_slope(x, y)
            slopes[stage][feat_key] = s

    return slopes

def get_life_stage(battery_id: str, cycle: int, total_cycle_life: float) -> str:
    """현재 수명 진행률로 early/mid/late 분류."""
    if not np.isfinite(total_cycle_life) or total_cycle_life <= 0:
        return "mid"
    frac = float(cycle) / float(total_cycle_life)
    if frac <= 0.30:
        return "early"
    if frac <= 0.70:
        return "mid"
    return "late"

def get_stage_slope(feature: str, stage: str) -> float:
    """stage slope 우선, 없으면 FEATURE_STATS(전역)로 fallback."""
    stage_slopes = compute_stage_slopes()
    s = float("nan")
    try:
        s = float(stage_slopes.get(stage, {}).get(feature, float("nan")))
    except Exception:
        s = float("nan")

    if np.isfinite(s):
        return float(s)

    # fallback to global stats
    if FEATURE_STATS is not None and feature in FEATURE_STATS:
        try:
            return float(FEATURE_STATS[feature].get("slope_rul_per_unit", 0.0))
        except Exception:
            return 0.0
    return 0.0
# -------------------------------------------------
# Expected band + anomaly evidence helpers (APM/Analytics)
# -------------------------------------------------
COHORT_OPTIONS = [
    ("전체(All)", "all"),
    ("주변온도 ≤ 10°C", "temp_le_10"),
    ("10–25°C", "temp_10_25"),
    ("≥ 25°C", "temp_ge_25"),
]

def apply_cohort_filter(df: pd.DataFrame, cohort_key: str) -> pd.DataFrame:
    if cohort_key == "all" or df is None:
        return df
    if "ambient_temp_c" not in df.columns:
        return df
    t = df["ambient_temp_c"].astype(float)
    if cohort_key == "temp_le_10":
        return df[t <= 10.0]
    if cohort_key == "temp_10_25":
        return df[(t > 10.0) & (t < 25.0)]
    if cohort_key == "temp_ge_25":
        return df[t >= 25.0]
    return df

@st.cache_resource
def compute_expected_band(metric_key: str, cohort_key: str):
    """cycle별 median/IQR expected band를 계산."""
    df = load_cycle_features()
    if df is None or metric_key not in df.columns:
        return None
    df = apply_cohort_filter(df, cohort_key)
    tmp = df[["cycle", metric_key]].dropna()
    if tmp.empty:
        return None
    g = tmp.groupby("cycle")[metric_key]
    med = g.median()
    q25 = g.quantile(0.25)
    q75 = g.quantile(0.75)
    return med, q25, q75

@st.cache_resource
def compute_capacity_pct_band(cohort_key: str):
    """capacity_mean을 배터리별 초기값으로 정규화한 Capacity(% of initial) expected band."""
    df = load_cycle_features()
    if df is None or "capacity_mean" not in df.columns:
        return None
    df = apply_cohort_filter(df, cohort_key)

    tmp = df[["battery_id", "cycle", "capacity_mean"]].dropna()
    if tmp.empty:
        return None

    # battery별 초기 capacity_mean (최소 cycle 기준)
    first = (
        tmp.sort_values(["battery_id", "cycle"])
        .groupby("battery_id")["capacity_mean"]
        .first()
        .rename("cap0")
    )
    tmp = tmp.join(first, on="battery_id")
    tmp = tmp[tmp["cap0"].astype(float) > 0]
    tmp["cap_pct"] = (tmp["capacity_mean"].astype(float) / tmp["cap0"].astype(float)) * 100.0

    g = tmp.groupby("cycle")["cap_pct"]
    med = g.median()
    q25 = g.quantile(0.25)
    q75 = g.quantile(0.75)
    return med, q25, q75


def robust_scale_from_iqr(q25: float, q75: float) -> float:
    iqr = float(q75 - q25)
    # normal distribution: IQR ≈ 1.349 * std
    return max(iqr / 1.349, 1e-9)

def compute_robust_z(value: float, med: float, q25: float, q75: float) -> float:
    if not (np.isfinite(value) and np.isfinite(med) and np.isfinite(q25) and np.isfinite(q75)):
        return float("nan")
    scale = robust_scale_from_iqr(q25, q75)
    return float((value - med) / scale)

def find_onset(z: np.ndarray, thresh: float, min_run: int = 3, direction: str = "abs"):
    """임계치 초과가 연속(min_run)으로 시작되는 최초 index 반환."""
    if z.size == 0:
        return None
    if direction == "pos":
        mask = np.isfinite(z) & (z >= thresh)
    elif direction == "neg":
        mask = np.isfinite(z) & (z <= -thresh)
    else:
        mask = np.isfinite(z) & (np.abs(z) >= thresh)

    run = 0
    for i, m in enumerate(mask):
        run = run + 1 if m else 0
        if run >= min_run:
            return i - min_run + 1
    return None

DRIVER_TAGS = {
    "thermal_stress": ("고온/열 스트레스", "열관리 점검(팬/냉각)·고온 구간 제한"),
    "temperature_mean": ("고온 노출", "냉각/통풍·고온 운행 제한"),
    "temp_rise_cycle": ("셀 발열 증가", "열 runaway 위험 체크·냉각 강화"),
    "eff_c_rate": ("고 C-rate(고부하)", "가속/급속충전 제한·부하 분산"),
    "current_max": ("고부하(충전/회생)", "피크 전류 제한·회생제동 설정 조정"),
    "current_min": ("고부하(방전)", "피크 방전 전류 제한·부하 분산"),
    "voltage_min": ("깊은 방전(DoD↑)", "최저 SoC 제한·운영전략 조정"),
    "dvdt_max_abs": ("전압 급변", "BMS 로깅/센서 점검·전력 프로파일 확인"),
    "dTdt_max": ("온도 급상승", "열관리/센서 점검·운행 제한"),
}

def top_driver_explanations(df_sel: pd.DataFrame, df_all: pd.DataFrame, cohort_key: str, cycle: int, features: List[str], topk: int = 3):
    """현재 cycle에서 driver 후보들의 cohort 대비 z-score를 계산해 Top-K 리턴.

    - df_sel: 선택 배터리의 cycle별 row들
    - df_all: 전체 배터리 cycle별 테이블 (cohort 산출용)
    """
    rows = []
    if df_sel is None or df_sel.empty or df_all is None or df_all.empty:
        return rows

    df_ref = apply_cohort_filter(df_all, cohort_key)
    df_ref_c = df_ref[df_ref["cycle"] == int(cycle)]
    df_sel_c = df_sel[df_sel["cycle"] == int(cycle)]

    if df_ref_c.empty or df_sel_c.empty:
        return rows

    for f in features:
        if f not in df_sel.columns or f not in df_ref_c.columns:
            continue
        try:
            v = float(df_sel_c.iloc[0][f])
        except Exception:
            continue

        vals = df_ref_c[f].astype(float).dropna()
        if len(vals) < 10:
            continue
        med = float(vals.median())
        q25 = float(vals.quantile(0.25))
        q75 = float(vals.quantile(0.75))
        z = compute_robust_z(v, med, q25, q75)
        tag, action = DRIVER_TAGS.get(f, (f, ""))
        rows.append({"feature": f, "value": v, "z": z, "tag": tag, "action": action})

    rows = sorted(rows, key=lambda r: abs(r.get("z", 0.0)), reverse=True)
    return rows[:topk]


@st.cache_resource
def compute_feature_ranges():
    df = load_cycle_features()
    out = {}
    if df is None:
        return out

    for feat_key, meta in SCENARIO_FEATURES.items():
        if feat_key not in df.columns:
            continue
        vals = df[feat_key].dropna()
        if len(vals) == 0:
            continue

        # 진짜 데이터 기준 min/max
        vmin = float(vals.min())
        vmax = float(vals.max())

        # 혹시라도 min==max인 경우(정말로 고정 피처인 경우) 대비
        if vmin == vmax:
            vmin -= meta["step"]
            vmax += meta["step"]

        out[feat_key] = {"min": vmin, "max": vmax}

    return out

FEATURE_RANGES = compute_feature_ranges()

def ensure_cycle_life_obs_in_record(rec: Dict[str, dict]) -> None:
    """
    rec dict에 관측 전체 수명(cycle_life_obs)이 없거나 NaN이면
    cycles 배열의 최댓값으로 채운다.
    """
    try:
        existing = float(rec.get("cycle_life_obs", float("nan")))
        if not math.isnan(existing):
            rec["cycle_life_obs"] = existing
            return
    except (TypeError, ValueError):
        pass

    cycles_arr = np.asarray(rec.get("cycles", []), dtype=float)
    if cycles_arr.size > 0:
        rec["cycle_life_obs"] = float(cycles_arr.max())
    else:
        rec["cycle_life_obs"] = float("nan")


# -------------------------------------------------
# runtime: ckpt + meta DB → dashboard records
# -------------------------------------------------
@st.cache_resource(show_spinner="Loading meta-learner & cycle DB...")
def load_meta_state(ckpt_path: str, eval_dataset: str = "from_ckpt"):
    return build_model_and_grouped(ckpt_path, eval_dataset=eval_dataset)


def build_runtime_records(meta_state, r_ratio: float = 0.3) -> Dict[str, dict]:
    """
    실시간 추론으로 dashboard용 records 생성.
    (ckpt 안의 test_bids 기준, 예: B0018, B0042, B0043)
    """
    (
        cfg,
        grouped,
        model,
        vecizer,
        meta_thetas,
        seq_scaler,
        sum_scaler,
        max_rul_train,
        test_bids,
    ) = meta_state

    records: Dict[str, dict] = {}

    for bid in test_bids:
        if bid not in grouped:
            continue
        g = grouped[bid]

        task = make_task_prefix(
            bid=bid,
            grouped=grouped,
            cfg=cfg,
            seq_scaler=seq_scaler,
            sum_scaler=sum_scaler,
            max_rul_train=max_rul_train,
            r_ratio=r_ratio,
            current_cycle=None,
        )
        pred_mean, pred_std = run_adapt_and_predict(
            cfg=cfg,
            model=model,
            vecizer=vecizer,
            meta_thetas=meta_thetas,
            task=task,
            max_rul_train=max_rul_train,
        )

        s_cyc = np.asarray(task["s_cycles_viz"], dtype=float)
        s_true = np.asarray(task["s_rul_viz"], dtype=float)
        q_cyc = np.asarray(task["q_cycles_viz"], dtype=float)
        q_true = np.asarray(task["q_rul_viz"], dtype=float)
        split_cycle = float(task["split_cycle"])

        pred_mean = np.asarray(pred_mean, dtype=float)
        pred_std = np.asarray(pred_std, dtype=float)

        cycles_full = np.concatenate([s_cyc, q_cyc])
        rul_true_full = np.concatenate([s_true, q_true])

        hist_nan = np.full_like(s_true, np.nan, dtype=float)
        pred_full = np.concatenate([hist_nan, pred_mean])
        std_full = np.concatenate([hist_nan, pred_std])

        mask = ~np.isnan(pred_full)
        if np.any(mask):
            diff = pred_full[mask] - rul_true_full[mask]
            rmse = float(np.sqrt(np.mean(diff**2)))
            mae = float(np.mean(np.abs(diff)))
        else:
            rmse = float("nan")
            mae = float("nan")

        cap_curve = g.get("capacity_curve", None)
        if cap_curve is not None and len(cap_curve) > 0:
            cap_arr = np.asarray(cap_curve, dtype=float)
            cap_init = float(cap_arr[0])
            cap_final = float(cap_arr[-1])
            cap_list = cap_arr.tolist()
        else:
            cap_init = float("nan")
            cap_final = float("nan")
            cap_list = None

        if cycles_full.size > 0:
            cycle_life_obs = float(cycles_full.max())
        else:
            cycle_life_obs = float("nan")

        rec = {
            "battery_id": bid,
            "cycles": cycles_full.tolist(),
            "rul_true": rul_true_full.tolist(),
            "rul_pred": pred_full.tolist(),
            "rul_std": std_full.tolist(),
            "split_cycle": split_cycle,
            "rmse": rmse,
            "mae": mae,
            "cap_init": cap_init,
            "cap_final": cap_final,
            "cycle_life_obs": cycle_life_obs,
        }
        if cap_list is not None:
            rec["capacity_curve"] = cap_list

        ensure_cycle_life_obs_in_record(rec)
        records[bid] = rec
    # (APM demo) 특정 배터리를 임의로 제외하지 않습니다.
    # 필요하면 아래 목록에 battery_id를 넣어 runtime records에서 제외할 수 있습니다.
    EXCLUDE_RUNTIME_BATTERIES = set()  # e.g., {"B9999"}
    for _bid in list(records.keys()):
        if _bid in EXCLUDE_RUNTIME_BATTERIES:
            records.pop(_bid, None)

    return records


def run_bmaml_once(r_ratio: float, note: str = ""):
    """
    버튼 클릭 시 호출:
    - meta-state 로드 (cache_resource)
    - runtime records 생성 후 session_state에 저장
    """
    loading_placeholder = st.empty()
    with loading_placeholder.container():
        st.markdown(
            f"""
            ### 🔧 meta-learner 초기화 중...
            - 체크포인트 로드
            - NASA 메타 DB에서 테스트 배터리 셀 로드
            - r_ratio={r_ratio:.2f} 기준 few-shot meta-adaptation 수행

            예측은 GPU/CPU 환경에 따라 시간이 걸릴 수 있습니다.
            로딩 중에는 화면을 가능한 한 건드리지 않는 것을 권장합니다.

            {note}
            """
        )
        if LOADING_GIF.exists():
            st.image(str(LOADING_GIF))

    try:
        with st.spinner("meta-learner & battery tasks 준비 중..."):
            meta_state = load_meta_state(str(CKPT_DEFAULT), eval_dataset="from_ckpt")
            records_rt = build_runtime_records(meta_state=meta_state, r_ratio=r_ratio)

        st.session_state["records"] = records_rt
        st.session_state["r_ratio"] = float(r_ratio)
        st.session_state["records_source"] = f"runtime (r_ratio={r_ratio:.2f})"
        loading_placeholder.empty()

    except Exception as e:
        loading_placeholder.empty()
        st.error(
            "runtime records 생성 실패.\n"
            f"- error: {e}"
        )
        st.stop()


# -------------------------------------------------
# App header
# -------------------------------------------------
st.set_page_config(
    page_title="Battery RUL Meta-Learning Dashboard (Real-time Inference)",
    layout="wide",
)

MODEL_TAG = "Fleet Battery Health APM / Analytics — Few-shot meta-learned RUL predictor (Transformer backbone)"
st.title("👁️🔋 실시간 리튬이온 배터리 RUL 예측 대시보드")
st.caption(MODEL_TAG)

# 시나리오 저장소 초기화
if "scenarios" not in st.session_state:
    st.session_state["scenarios"] = []

if "scenario_message" not in st.session_state:
    st.session_state["scenario_message"] = None

# -------------------------------------------------
# 0. 처음 진입: precomputed DEFAULT_R_RATIO 우선 사용
# -------------------------------------------------
if "records" not in st.session_state:
    export_root = BMAML_DIR
    loader = PrecomputedRULLoader(export_root)

    target_r = float(DEFAULT_R_RATIO)
    new_records = {}

    # 0) 초기에는 EXPORT_TEST_BATTERIES 기준으로 시도
    candidate_bids = sorted(EXPORT_TEST_BATTERIES)

    # 1차: PrecomputedRULLoader 포맷 (예: B0018_r0.25.json)
    for bid in candidate_bids:
        if loader.has_precomputed(bid, target_r):
            rec = loader.load(bid, target_r)
            if rec is not None:
                bid_loaded = str(rec.get("battery_id", bid))
                ensure_cycle_life_obs_in_record(rec)
                new_records[bid_loaded] = rec

    # 2차: 예전 포맷 (battery_B0018_r0p25.json) fallback
    if not new_records:
        pre_idx = load_precomputed_index(export_root)
        if pre_idx:
            for bid, r_map in pre_idx.items():
                if target_r not in r_map:
                    continue
                path = r_map[target_r]
                with open(path, "r") as f:
                    rec = json.load(f)
                bid_loaded = str(rec.get("battery_id", bid))
                ensure_cycle_life_obs_in_record(rec)
                new_records[bid_loaded] = rec

    if new_records:
        st.session_state["records"] = new_records
        st.session_state["r_ratio"] = target_r
        st.session_state["records_source"] = f"precomputed JSON (r_ratio={target_r:.2f})"
    else:
        # 진짜로 해당 r_ratio에 대한 precomputed가 하나도 없을 때만 에러
        st.error(
            f"DEFAULT_R_RATIO={DEFAULT_R_RATIO:.2f} 에 대한 precomputed JSON을 찾을 수 없습니다.\n"
            "export_rul_dashboard_data_meta_fixed.py 를 이 r_ratio로 먼저 돌려 주세요."
        )
        st.stop()

st.sidebar.header("1) 모델 설정")

# 체크포인트 없어도 precomputed는 쓸 수 있게
ckpt_exists = CKPT_DEFAULT.exists()

if not RUNTIME_AVAILABLE:
    st.sidebar.info(
        "torch 등 런타임 의존성이 없어서 **'빠른 로드(precomputed)' 모드만** 사용할 수 있습니다.\n"
        f"(import error: {_RUNTIME_IMPORT_ERROR})"
    )
if not ckpt_exists:
    st.sidebar.warning(
        f"checkpoint not found:\n{CKPT_DEFAULT}\n"
        "→ 실시간 재적응은 사용할 수 없고, 아래 '빠른 로드 (precomputed)'만 사용 가능합니다."
    )

# 🔽 슬라이더 기본값을 0.15~0.40 사이로 클램프
slider_default = float(
    np.clip(st.session_state.get("r_ratio", DEFAULT_R_RATIO), 0.15, 0.40)
)

r_ratio = st.sidebar.slider(
    "초기 적응 비율 (r_ratio)",
    min_value=0.15,
    max_value=0.40,
    value=slider_default,
    step=0.05,
    help="전체 사이클 중 어느 비율까지를 초기 데이터로 사용해 실시간 추론을 할지 정합니다.",
)

# ❗ 이 문구 유지
st.sidebar.caption(
    "※ 더 적은 초기 데이터만 보고 예측해 보고 싶다면 비율을 조금 낮춰 보세요 (예: 0.25 → 0.20)."
)

# 1) 재추론 버튼
init_clicked = st.sidebar.button(
    "모델 초기화 및 재추론 수행",
    disabled=(not ckpt_exists) or (not RUNTIME_AVAILABLE),
)

# 재추론 안내 문구
st.sidebar.caption(
    "재추론은 1분 정도 소요되나, 빠른 로드를 원한다면 아래 버튼을 클릭해주세요."
)

# 2) 빠른 로드 버튼 (precomputed)
fast_load_clicked = st.sidebar.button("빠른 로드 (precomputed)")

if init_clicked:
    st.session_state["force_recompute"] = True
    run_bmaml_once(r_ratio=r_ratio)

elif fast_load_clicked:
    st.session_state["force_recompute"] = False

    export_root = BMAML_DIR
    loader = PrecomputedRULLoader(export_root)
    target_r = float(r_ratio)

    base_records = st.session_state.get("records", {})
    candidate_bids = sorted(base_records.keys()) if base_records else sorted(EXPORT_TEST_BATTERIES)

    new_records = {}

    # 1차: PrecomputedRULLoader 포맷 (예: B0018_r0.20.json)
    for bid in candidate_bids:
        if loader.has_precomputed(bid, target_r):
            rec = loader.load(bid, target_r)
            if rec is not None:
                bid_loaded = str(rec.get("battery_id", bid))
                ensure_cycle_life_obs_in_record(rec)
                new_records[bid_loaded] = rec

    # 2차: 예전 포맷 (battery_B0018_r0p20.json) fallback
    if not new_records:
        pre_idx = load_precomputed_index(export_root)
        if pre_idx:
            for bid, r_map in pre_idx.items():
                if target_r not in r_map:
                    continue
                path = r_map[target_r]
                with open(path, "r") as f:
                    rec = json.load(f)
                bid_loaded = str(rec.get("battery_id", bid))
                ensure_cycle_life_obs_in_record(rec)
                new_records[bid_loaded] = rec

    if new_records:
        st.session_state["records"] = new_records
        st.session_state["r_ratio"] = target_r
        st.session_state["records_source"] = f"precomputed JSON (r_ratio={target_r:.2f})"

        loaded_bids = ", ".join(sorted(new_records.keys()))
        st.sidebar.success(
            f"r_ratio={target_r:.2f} 기준 미리 계산된 RUL 결과를 로드했습니다.\n"
            f"적용된 배터리: {loaded_bids}"
        )
    else:
        st.sidebar.error(
            f"r_ratio={target_r:.2f} 에 대한 미리 계산된 JSON을 찾지 못했습니다.\n"
            "export 경로나 파일명을 한 번 확인해 주세요."
        )
else:
    st.session_state["force_recompute"] = False


# 아직 records가 전혀 없으면 안내 후 종료
if "records" not in st.session_state or not st.session_state["records"]:
    st.info(
        "왼쪽에서 **모델 초기화 및 재추론 수행** 버튼을 눌러 데이터를 불러와 주세요.\n\n"
        "예측/추론은 약 **1분 내외**로 소요됩니다."
    )
    st.stop()

records = st.session_state["records"]
current_r_ratio = st.session_state.get("r_ratio", r_ratio)
records_source = st.session_state.get("records_source", "precomputed exports")

# ----------------- Sidebar: Battery & cursor -----------------
st.sidebar.header("2) 테스트 배터리 및 사이클 선택")

all_bids = sorted(records.keys())
# 초기 적응 비율 0.25에서 사용한 것과 동일한 배터리 셋을 항상 우선 배치
preferred_bids = ["B0018", "B0033", "B0043", "B0055"]
all_bids = [b for b in preferred_bids if b in all_bids] + [
    b for b in all_bids if b not in preferred_bids
]

selected_bid = st.sidebar.selectbox(
    "Battery ID (NASA test cells)",
    all_bids,
)

rec = records[selected_bid]
cycles = np.asarray(rec["cycles"], dtype=float)
rul_true = np.asarray(rec["rul_true"], dtype=float)
rul_pred = np.asarray(rec["rul_pred"], dtype=float)
rul_std = np.asarray(rec.get("rul_std", [np.nan] * len(rul_pred)), dtype=float)
capacity_curve = np.asarray(rec.get("capacity_curve", []), dtype=float)
has_capacity = capacity_curve.size == cycles.size

split_cycle = rec.get("split_cycle", float(cycles[int(len(cycles) * 0.5)]))
split_idx = int(np.argmin(np.abs(cycles - split_cycle)))

hist_cycles_all = cycles[: split_idx + 1]
hist_rul_all = rul_true[: split_idx + 1]

fut_cycles = cycles[split_idx + 1 :]
fut_rul_true = rul_true[split_idx + 1 :]
fut_rul_pred = rul_pred[split_idx + 1 :]
fut_rul_std = rul_std[split_idx + 1 :]

min_cycle = int(cycles.min())
max_cycle = int(cycles.max())

# 선택 배터리 바뀌면 내부 상태 리셋
if "selected_bid" not in st.session_state:
    st.session_state.selected_bid = selected_bid
if st.session_state.selected_bid != selected_bid:
    st.session_state.selected_bid = selected_bid
    st.session_state["play_cycle"] = max(min_cycle, 1)

# 🔁 Auto-play 체크박스
auto_play = st.sidebar.checkbox("사이클 자동 재생", key="auto_play", value=False)

# 🎚 슬라이더 (위젯용 key는 따로)
current_cycle_slider = st.sidebar.slider(
    "현재 사이클 (수동 모드)",
    min_value=min_cycle,
    max_value=max_cycle,
    step=1,
    value=st.session_state.get("play_cycle", max(min_cycle, 1)),
    key="current_cycle_slider",
    disabled=st.session_state.get("auto_play", False),
)
st.sidebar.caption(
    "※ r_ratio 이후 구간(예: 25% 이후)으로 옮기면, "
    "모델이 예측한 RUL 구간부터 바로 볼 수 있습니다."
)

# 내부 play_cycle 초기화
if "play_cycle" not in st.session_state:
    st.session_state["play_cycle"] = current_cycle_slider

# 현재 run에서 사용할 실제 cycle 값 결정
if st.session_state.get("auto_play", False):
    current_cycle = int(st.session_state["play_cycle"])
else:
    current_cycle = int(current_cycle_slider)
    st.session_state["play_cycle"] = current_cycle

# metrics
st.sidebar.markdown("---")
rmse = rec.get("rmse", float("nan"))
mae = rec.get("mae", float("nan"))

st.sidebar.subheader("모델 예측 오차 지표")

# 전체 future region 고정 값만 사이드바에 표시
if not math.isnan(rmse):
    st.sidebar.metric("RMSE (future region)", f"{rmse:.2f}")
if not math.isnan(mae):
    st.sidebar.metric("MAE (future region)", f"{mae:.2f}")


# -------------------------------------------------
# 공통 값들 (cycles / RUL 기준 계산)
# -------------------------------------------------
all_true = rul_true
idx_slider = int(np.argmin(np.abs(cycles - current_cycle)))

initial_rul = float(all_true.max()) if all_true.size > 0 else 1.0
current_true_rul = float(all_true[idx_slider]) if all_true.size > 0 else 0.0

in_future_region = fut_cycles.size > 0 and current_cycle >= float(fut_cycles.min())

# 관측(ground-truth) 잔여수명: 전체 구간에서 항상 정의됨
current_obs_rul = current_true_rul

# 예측 잔여수명: prefix 이후(=future region)에서만 정의
if in_future_region and fut_rul_pred.size > 0:
    idx_pred = int(np.argmin(np.abs(fut_cycles - current_cycle)))
    current_pred_rul = float(fut_rul_pred[idx_pred])
else:
    current_pred_rul = float("nan")

# 🔹 현재 사이클 절대 오차 (그래프 아래에서 사용할 값)
current_abs_err = None
if (
    fut_cycles.size > 0
    and current_cycle >= float(fut_cycles.min())
    and fut_rul_pred.size > 0
):
    idx_pred_curr = int(np.argmin(np.abs(fut_cycles - current_cycle)))
    current_abs_err = float(
        abs(fut_rul_pred[idx_pred_curr] - fut_rul_true[idx_pred_curr])
    )


# capacity info
cap_init = rec.get("cap_init", float("nan"))
cap_final = rec.get("cap_final", float("nan"))
cycle_life_obs = rec.get("cycle_life_obs", float("nan"))
# 🔹 NASA 전체 DB 기준 관측 수명 (예: 132 cycles)으로 다시 계산
total_cycle_life = cycle_life_obs
feat_df_all = load_cycle_features()
if feat_df_all is not None:
    mask_b = feat_df_all["battery_id"] == str(selected_bid)
    sub_b = feat_df_all.loc[mask_b]
    if not sub_b.empty:
        total_cycle_life = float(sub_b["cycle"].max())
life_stage = get_life_stage(str(selected_bid), int(current_cycle), float(total_cycle_life))

if has_capacity:
    current_cap = float(capacity_curve[idx_slider])
else:
    current_cap = float("nan")

# Usage (scenario builder와 공유)
cycles_per_day = st.session_state.get("cycles_per_day", DEFAULT_CYCLES_PER_DAY)

# 파이프라인에서 계산된 SoH 읽기 (nasa_features_rul.csv 기반)
cycle_ctx = CycleFeatureContext(selected_bid, current_cycle)
pipeline_soh = cycle_ctx.soh

if not math.isnan(pipeline_soh):
    soh_pct = max(0.0, min(100.0, pipeline_soh * 100.0))
else:
    soh_pct = float("nan")

# SoH 기준 상태 등급 (현재는 직접 사용하지 않지만 남겨둠)
if not math.isnan(soh_pct):
    if soh_pct >= 90.0:
        health_status, health_emoji, color = "양호", "🟢", "#2ca02c"
    elif soh_pct >= 80.0:
        health_status, health_emoji, color = "점검 권장", "🟡", "#ff7f0e"
    else:
        health_status, health_emoji, color = "교체/정비 필요", "🔴", "#d62728"
else:
    health_status, health_emoji, color = "정보 부족", "⚪️", "#888888"

# 1사이클당 주행거리(km) 가정
km_per_cycle = st.session_state.get("km_per_cycle", 400.0)  # 기본 400 km/사이클 (참고용 가정)
# 일/거리 환산은 예측 가능하면 예측 RUL, 아니면 관측 RUL을 사용(표기에서 구분)
_rul_for_days = current_pred_rul if not math.isnan(current_pred_rul) else current_obs_rul
remaining_days = _rul_for_days / max(cycles_per_day, 1e-6)
remaining_km = _rul_for_days * km_per_cycle

# =================================================
# 2x2 Layout
# =================================================
top_left, top_right = st.columns([2.0, 2.0])
bottom_left, bottom_right = st.columns([2.0, 2.0])
# =================================================
# TOP-LEFT: Battery info & health + GIF
# =================================================
with top_left:
    st.markdown(
        f"<div style='font-size:28px; font-weight:700;'>배터리 {selected_bid}</div>",
        unsafe_allow_html=True,
    )

    # 🔹 상태 계산 (공통) – 카드 / GIF 둘 다 사용
    status_basis = "예측 기반" if (initial_rul > 0 and not math.isnan(current_pred_rul)) else "관측 기반"
    rul_for_status = current_pred_rul if (initial_rul > 0 and not math.isnan(current_pred_rul)) else current_obs_rul

    if initial_rul > 0 and not math.isnan(rul_for_status):
        rul_pct = max(0.0, min(100.0, 100.0 * float(rul_for_status) / float(initial_rul)))

        if rul_pct >= 60.0:
            status_txt, status_emoji, status_color = "양호", "🟢", "#2ca02c"
        elif rul_pct >= 30.0:
            status_txt, status_emoji, status_color = "주의", "🟡", "#ff7f0e"
        else:
            status_txt, status_emoji, status_color = "교체/정비 고려", "🔴", "#d62728"
    else:
        rul_pct = float("nan")
        status_txt, status_emoji, status_color = "정보 부족", "⚪", "#999999"

# 🔹 4개 카드: 현재 사이클 / 관측 RUL / 예측 RUL / 상태(작은 카드)
    h1, h2, h3, h4 = st.columns(4)

    with h1:
        st.metric("현재 사이클", current_cycle)

    with h2:
        if not math.isnan(current_obs_rul):
            st.metric("관측 잔여 수명 (사이클)", f"{current_obs_rul:.1f}")
        else:
            st.metric("관측 잔여 수명 (사이클)", "데이터 없음")

    with h3:
        if not math.isnan(current_pred_rul):
            st.metric("예측 잔여 수명 (사이클)", f"{current_pred_rul:.1f}")
        else:
            st.metric("예측 잔여 수명 (사이클)", "예측 없음")

    with h4:
        # st.metric 대신 작은 커스텀 카드 (글자 확 줄임)
        if not math.isnan(rul_pct):
            st.markdown(
                f"""
                <div style='border-radius:10px; border:1px solid #e0e0e0;
                            padding:6px 10px; text-align:center;'>
                    <div style='font-size:11px; color:#777; margin-bottom:2px;'>현재 상태 ({status_basis})</div>
                    <div style='font-size:14px; font-weight:600; color:{status_color};'>
                        {status_emoji} {status_txt} ({rul_pct:.1f}%)
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style='border-radius:10px; border:1px solid #e0e0e0;
                            padding:6px 10px; text-align:center;'>
                    <div style='font-size:11px; color:#777; margin-bottom:2px;'>현재 상태</div>
                    <div style='font-size:14px; font-weight:600; color:#999;'>정보 부족</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

# 🔹 GIF도 같은 rul_pct 기준으로 선택 (예전 로직 복구)
    gif_to_show = None
    if not math.isnan(rul_pct):
        if rul_pct >= 60.0 and HEALTH_HIGH_GIF.exists():
            gif_to_show = HEALTH_HIGH_GIF
        elif rul_pct >= 30.0 and HEALTH_MED_GIF.exists():
            gif_to_show = HEALTH_MED_GIF
        elif HEALTH_LOW_GIF.exists():
            gif_to_show = HEALTH_LOW_GIF

    if gif_to_show is not None:
        st.image(str(gif_to_show))
        st.markdown(
            "<div style='font-size:11px; color:#888; text-align:center; margin-top:4px;'>"
            "위 GIF는 이해를 돕기 위한 예시 이미지입니다."
            "</div>",
            unsafe_allow_html=True,
        )




# =================================================
# Monitoring snapshot (current cycle)
# =================================================
st.markdown("#### Degradation snapshot (current cycle)")
if cycle_ctx.row is None:
    st.caption("※ nasa_features_rul.csv 에서 해당 배터리/사이클 피처를 찾지 못했습니다.")
else:
    row = cycle_ctx.row
    df_feat = load_cycle_features()
    cap0 = None
    if df_feat is not None and "capacity_mean" in df_feat.columns:
        sub0 = df_feat[df_feat["battery_id"] == str(selected_bid)].sort_values("cycle")
        if not sub0.empty:
            try:
                cap0 = float(sub0.iloc[0]["capacity_mean"])
            except Exception:
                cap0 = None

    def _get_float(key):
        try:
            if row is None or key not in row:
                return float("nan")
            return float(row[key])
        except Exception:
            return float("nan")

    soh_v = _get_float("soh")
    cap_v = _get_float("capacity_mean")
    cap_pct = (cap_v / cap0 * 100.0) if (cap0 is not None and cap0 > 0 and not math.isnan(cap_v)) else float("nan")

    dcr_v = _get_float("dcr")
    imp_v = _get_float("impedance_sum")
    temp_v = _get_float("temperature_mean")
    stress_v = _get_float("thermal_stress")
    dcr_g = _get_float("dcr_growth")
    imp_g = _get_float("impedance_growth")

    r1c1, r1c2, r1c3 = st.columns(3)
    with r1c1:
        st.metric("SoH", f"{soh_v*100:.1f}%" if not math.isnan(soh_v) else "데이터 없음")
    with r1c2:
        if not math.isnan(cap_pct):
            st.metric("Capacity (vs. initial)", f"{cap_pct:.1f}%")
        else:
            st.metric("Capacity (vs. initial)", "데이터 없음")
    with r1c3:
        st.metric("DCR", f"{dcr_v:.4f} Ω" if not math.isnan(dcr_v) else "데이터 없음")

    r2c1, r2c2, r2c3 = st.columns(3)
    with r2c1:
        st.metric("Impedance sum", f"{imp_v:.4f} Ω" if not math.isnan(imp_v) else "데이터 없음")
    with r2c2:
        st.metric("Temp mean", f"{temp_v:.1f} °C" if not math.isnan(temp_v) else "데이터 없음")
    with r2c3:
        st.metric("Thermal stress", f"{stress_v:.3f}" if not math.isnan(stress_v) else "데이터 없음")

    # 작은 텍스트: 악화 속도(있을 때만)
    if (not math.isnan(dcr_g)) or (not math.isnan(imp_g)):
        st.caption(
            "악화 속도: "
            + (f"DCR_growth={dcr_g:+.4f}  " if not math.isnan(dcr_g) else "")
            + (f"Impedance_growth={imp_g:+.4f}" if not math.isnan(imp_g) else "")
        )

    # # 🔹 관측 전체 수명은 각주로만 표시 (너 보려고만)
    # if not math.isnan(total_cycle_life):
    #     st.caption(f"※ 관측 전체 수명 (사이클): {total_cycle_life:.0f}")

# =================================================
# TOP-RIGHT: RUL trajectory
# =================================================
with top_right:
    st.markdown("### Battery remaining life trajectory (RUL)")
    fig = go.Figure()

    # 1) True RUL: 전체 구간
    fig.add_trace(
        go.Scatter(
            x=cycles,
            y=rul_true,
            mode="lines",
            name="실제 RUL",
            line=dict(color="rgba(0,0,0,0.5)", dash="dash"),
        )
    )

    # 2) Observed RUL (history): current cycle 이전만
    mask_hist = hist_cycles_all <= float(current_cycle)
    fig.add_trace(
        go.Scatter(
            x=hist_cycles_all[mask_hist],
            y=hist_rul_all[mask_hist],
            mode="lines",
            name="관측 RUL (과거)",
            line=dict(color="rgb(70,70,70)"),
        )
    )

    # 3) current cycle 이후 예측 + 불확실성
    if in_future_region and fut_cycles.size > 0:
        mask_future = fut_cycles >= float(current_cycle)
        fut_x = fut_cycles[mask_future]
        fut_pred_y = fut_rul_pred[mask_future]
        fut_std_y = fut_rul_std[mask_future]

        if fut_x.size > 0:
            # 불확실성 밴드 (±2σ)
            if fut_std_y.size == fut_pred_y.size and not np.all(np.isnan(fut_std_y)):
                upper = fut_pred_y + 2.0 * fut_std_y
                lower = np.maximum(0.0, fut_pred_y - 2.0 * fut_std_y)

                fig.add_trace(
                    go.Scatter(
                        x=np.concatenate([fut_x, fut_x[::-1]]),
                        y=np.concatenate([upper, lower[::-1]]),
                        fill="toself",
                        fillcolor="rgba(31,119,180,0.25)",
                        line=dict(color="rgba(0,0,0,0)"),
                        hoverinfo="skip",
                        showlegend=True,
                        name="불확실성 (±2σ)",
                    )
                )

            # 예측 RUL 곡선
            fig.add_trace(
                go.Scatter(
                    x=fut_x,
                    y=fut_pred_y,
                    mode="lines",
                    name="예측 RUL",
                    line=dict(color="rgb(214,39,40)"),
                )
            )

    # ---------------------------
    # Split line (current cycle)
    # ---------------------------
    fig.add_vline(
        x=float(current_cycle),
        line_width=2,
        line_dash="dash",
        line_color="red",
        annotation_text="Split (current)",
        annotation_position="top right",
    )

    # ---------------------------
    # EOL line: 첫 RUL<=0 위치
    # ---------------------------
    if cycles.size > 0 and rul_true.size == cycles.size:
        eol_idxs = np.where(rul_true <= 0)[0]
        if eol_idxs.size > 0:
            eol_cycle = float(cycles[eol_idxs[0]])
            fig.add_vline(
                x=eol_cycle,
                line_width=2,
                line_dash="dash",
                line_color="orange",
                annotation_text="EOL (RUL=0)",
                annotation_position="top right",
            )

    fig.update_layout(
        xaxis_title="사이클 인덱스",
        yaxis_title="RUL (잔여 수명)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=40, r=20, t=12, b=30),
        height=360,
    )
    st.plotly_chart(fig, use_container_width=True)

        # 🔹 그래프 아래에 작은 텍스트로 RMSE/MAE/절대오차 표시
    mask_dyn = (~np.isnan(rul_pred)) & (cycles <= float(current_cycle))
    rmse_dyn = mae_dyn = None
    if np.any(mask_dyn):
        diff_dyn = rul_pred[mask_dyn] - rul_true[mask_dyn]
        rmse_dyn = float(np.sqrt(np.mean(diff_dyn**2)))
        mae_dyn = float(np.mean(np.abs(diff_dyn)))

    # 값이 있을 때만 출력
    if (rmse_dyn is not None) and (mae_dyn is not None) and (current_abs_err is not None):
        c_rmse, c_mae, c_err = st.columns(3)

        with c_rmse:
            st.markdown(
                f"""
                <div style='background-color:#f3e5f5; padding:6px 10px;
                            border-radius:6px; text-align:center;'>
                  <div style='font-size:11px; color:#6a1b9a;'>RMSE (현재까지 예측구간)</div>
                  <div style='font-size:16px; font-weight:600; color:#311b92;'>
                    {rmse_dyn:.2f}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c_mae:
            st.markdown(
                f"""
                <div style='background-color:#e3f2fd; padding:6px 10px;
                            border-radius:6px; text-align:center;'>
                  <div style='font-size:11px; color:#1565c0;'>MAE (현재까지 예측구간)</div>
                  <div style='font-size:16px; font-weight:600; color:#0d47a1;'>
                    {mae_dyn:.2f}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with c_err:
            st.markdown(
                f"""
                <div style='background-color:#fff3e0; padding:6px 10px;
                            border-radius:6px; text-align:center;'>
                  <div style='font-size:11px; color:#ef6c00;'>현재 사이클 절대 오차</div>
                  <div style='font-size:16px; font-weight:600; color:#e65100;'>
                    {current_abs_err:.2f}
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )



# =================================================
# Degradation monitoring (timeline + comparison)
# =================================================
with st.expander("📈 Degradation monitoring (selected battery)", expanded=True):
    df_feat = load_cycle_features()
    if df_feat is None:
        st.info("analysis/nasa_features_rul.csv 를 찾지 못해 모니터링 타임라인을 표시할 수 없습니다.")
    else:
        cohort_label_mon = st.selectbox("Reference cohort (expected band)", [c[0] for c in COHORT_OPTIONS], index=0, key="monitor_cohort")
        cohort_key_mon = dict(COHORT_OPTIONS)[cohort_label_mon]

        if cohort_key_mon != "all":
            st.warning(
                "⚠️ NASA 데이터는 온도 조건과 수명 단계가 confounding될 수 있어, "
                "온도 cohort로 expected band를 만들면 편향될 수 있습니다. (참고용으로만 사용)"
            )

        online_mode_mon = st.checkbox(
            "온라인 모니터링 모드 (현재 cycle 이후 숨김)",
            value=True,
            help="실시간 운영을 흉내 내기 위해, 현재 선택한 cycle 이후의 관측값을 숨깁니다.",
            key="mon_online_mode",
        )
        overlay_band_mon = st.checkbox(
            "Expected band(중앙값/IQR) 오버레이",
            value=True,
            help="reference cohort(기대 범위)의 중앙값/사분위 밴드를 그래프 위에 표시합니다.",
            key="mon_overlay_band",
        )

        df_b = df_feat[df_feat["battery_id"] == str(selected_bid)].sort_values("cycle")
        if online_mode_mon:
            df_b = df_b[df_b["cycle"] <= int(current_cycle)].copy()
        if df_b.empty:
            st.info("해당 배터리의 cycle feature가 없습니다.")
        else:
            # Capacity를 %로 정규화(초기 대비)
            cap_pct_series = None
            if "capacity_mean" in df_b.columns:
                try:
                    cap0 = float(df_b.iloc[0]["capacity_mean"])
                    if cap0 > 0:
                        cap_pct_series = (df_b["capacity_mean"].astype(float) / cap0) * 100.0
                except Exception:
                    cap_pct_series = None

            fig_mon = make_subplots(
                rows=3,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                subplot_titles=(
                    "SoH / Capacity (normalized)",
                    "Impedance & DCR",
                    "Thermal signals",
                ),
            )

            x = df_b["cycle"].astype(float).values

            # Row 1: SoH + Capacity%
            if "soh" in df_b.columns:
                fig_mon.add_trace(
                    go.Scatter(x=x, y=df_b["soh"].astype(float) * 100.0, mode="lines", name="SoH (%)"),
                    row=1, col=1
                )
            if cap_pct_series is not None:
                fig_mon.add_trace(
                    go.Scatter(x=x, y=cap_pct_series, mode="lines", name="Capacity (% of initial)"),
                    row=1, col=1
                )
            # Expected band overlay (Capacity % of initial)
            if overlay_band_mon and cap_pct_series is not None:
                band = compute_capacity_pct_band(cohort_key_mon)
                if band is not None:
                    med, q25, q75 = band
                    x_band = np.asarray([c for c in x if int(c) in med.index], dtype=float)
                    if x_band.size > 0:
                        y_med = np.asarray([float(med.loc[int(c)]) for c in x_band], dtype=float)
                        y_q25 = np.asarray([float(q25.loc[int(c)]) for c in x_band], dtype=float)
                        y_q75 = np.asarray([float(q75.loc[int(c)]) for c in x_band], dtype=float)

                        fig_mon.add_trace(
                            go.Scatter(x=x_band, y=y_q75, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip", name="cap_q75"),
                            row=1, col=1
                        )
                        fig_mon.add_trace(
                            go.Scatter(x=x_band, y=y_q25, mode="lines", fill="tonexty", line=dict(width=0), opacity=0.15,
                                       hoverinfo="skip", name="Expected IQR (Cap%)"),
                            row=1, col=1
                        )
                        fig_mon.add_trace(
                            go.Scatter(x=x_band, y=y_med, mode="lines", line=dict(dash="dash"), name="Expected median (Cap%)"),
                            row=1, col=1
                        )

            # Row 2: impedance_sum + dcr
            if "impedance_sum" in df_b.columns:
                fig_mon.add_trace(
                    go.Scatter(x=x, y=df_b["impedance_sum"].astype(float), mode="lines", name="Impedance sum (Ω)"),
                    row=2, col=1
                )
            if "dcr" in df_b.columns:
                fig_mon.add_trace(
                    go.Scatter(x=x, y=df_b["dcr"].astype(float), mode="lines", name="DCR (Ω)"),
                    row=2, col=1
                )
            # Expected band overlay (DCR)
            if overlay_band_mon and "dcr" in df_b.columns:
                band = compute_expected_band("dcr", cohort_key_mon)
                if band is not None:
                    med, q25, q75 = band
                    x_band = np.asarray([c for c in x if int(c) in med.index], dtype=float)
                    if x_band.size > 0:
                        y_med = np.asarray([float(med.loc[int(c)]) for c in x_band], dtype=float)
                        y_q25 = np.asarray([float(q25.loc[int(c)]) for c in x_band], dtype=float)
                        y_q75 = np.asarray([float(q75.loc[int(c)]) for c in x_band], dtype=float)

                        fig_mon.add_trace(
                            go.Scatter(x=x_band, y=y_q75, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip", name="dcr_q75"),
                            row=2, col=1
                        )
                        fig_mon.add_trace(
                            go.Scatter(x=x_band, y=y_q25, mode="lines", fill="tonexty", line=dict(width=0), opacity=0.15,
                                       hoverinfo="skip", name="Expected IQR (DCR)"),
                            row=2, col=1
                        )
                        fig_mon.add_trace(
                            go.Scatter(x=x_band, y=y_med, mode="lines", line=dict(dash="dash"), name="Expected median (DCR)"),
                            row=2, col=1
                        )
            if "impedance_growth" in df_b.columns:
                fig_mon.add_trace(
                    go.Scatter(x=x, y=df_b["impedance_growth"].astype(float), mode="lines", name="Impedance growth"),
                    row=2, col=1
                )
            if "dcr_growth" in df_b.columns:
                fig_mon.add_trace(
                    go.Scatter(x=x, y=df_b["dcr_growth"].astype(float), mode="lines", name="DCR growth"),
                    row=2, col=1
                )

            # Row 3: thermal signals
            if "temperature_mean" in df_b.columns:
                fig_mon.add_trace(
                    go.Scatter(x=x, y=df_b["temperature_mean"].astype(float), mode="lines", name="Temp mean (°C)"),
                    row=3, col=1
                )
            if "temp_rise_cycle" in df_b.columns:
                fig_mon.add_trace(
                    go.Scatter(x=x, y=df_b["temp_rise_cycle"].astype(float), mode="lines", name="Temp rise/cycle"),
                    row=3, col=1
                )
            if "thermal_stress" in df_b.columns:
                fig_mon.add_trace(
                    go.Scatter(x=x, y=df_b["thermal_stress"].astype(float), mode="lines", name="Thermal stress"),
                    row=3, col=1
                )
            # Expected band overlay (Thermal stress)
            if overlay_band_mon and "thermal_stress" in df_b.columns:
                band = compute_expected_band("thermal_stress", cohort_key_mon)
                if band is not None:
                    med, q25, q75 = band
                    x_band = np.asarray([c for c in x if int(c) in med.index], dtype=float)
                    if x_band.size > 0:
                        y_med = np.asarray([float(med.loc[int(c)]) for c in x_band], dtype=float)
                        y_q25 = np.asarray([float(q25.loc[int(c)]) for c in x_band], dtype=float)
                        y_q75 = np.asarray([float(q75.loc[int(c)]) for c in x_band], dtype=float)

                        fig_mon.add_trace(
                            go.Scatter(x=x_band, y=y_q75, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip", name="ts_q75"),
                            row=3, col=1
                        )
                        fig_mon.add_trace(
                            go.Scatter(x=x_band, y=y_q25, mode="lines", fill="tonexty", line=dict(width=0), opacity=0.12,
                                       hoverinfo="skip", name="Expected IQR (Thermal)"),
                            row=3, col=1
                        )
                        fig_mon.add_trace(
                            go.Scatter(x=x_band, y=y_med, mode="lines", line=dict(dash="dash"), name="Expected median (Thermal)"),
                            row=3, col=1
                        )

            # current cycle marker
            for r in [1, 2, 3]:
                fig_mon.add_vline(x=float(current_cycle), line_width=1, line_dash="dash", line_color="red", row=r, col=1)

            fig_mon.update_layout(
                height=560,
                margin=dict(l=30, r=10, t=40, b=30),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                xaxis3_title="Cycle",
            )
            st.plotly_chart(fig_mon, use_container_width=True)

            # -------------------------------------------------
            # Anomaly report: expected band 대비 이탈 증빙 + '왜?'(driver) 연결
            # -------------------------------------------------
            st.markdown("#### 🧪 Anomaly report (expected vs observed)")

            cycles_arr = df_b["cycle"].astype(int).values

            # 1) DCR (fault-like) 이탈
            z_dcr = None
            dcr_max_z = float("nan")
            dcr_onset_cycle = None
            if "dcr" in df_b.columns:
                band = compute_expected_band("dcr", cohort_key_mon)
                if band is not None:
                    med, q25, q75 = band
                    z_list = []
                    for c, v in zip(cycles_arr, df_b["dcr"].astype(float).values):
                        if c in med.index:
                            z_list.append(
                                compute_robust_z(
                                    float(v),
                                    float(med.loc[c]),
                                    float(q25.loc[c]),
                                    float(q75.loc[c]),
                                )
                            )
                        else:
                            z_list.append(float("nan"))
                    z_dcr = np.asarray(z_list, dtype=float)
                    if np.any(np.isfinite(z_dcr)):
                        dcr_max_z = float(np.nanmax(z_dcr))
                        onset_i = find_onset(z_dcr, thresh=3.0, min_run=2, direction="pos")
                        if onset_i is not None:
                            dcr_onset_cycle = int(cycles_arr[onset_i])

            # 2) Capacity(% of initial) (accelerated) 이탈
            z_cap = None
            cap_min_z = float("nan")
            cap_onset_cycle = None
            if cap_pct_series is not None:
                band = compute_capacity_pct_band(cohort_key_mon)
                if band is not None:
                    med, q25, q75 = band
                    z_list = []
                    for c, v in zip(cycles_arr, cap_pct_series.astype(float).values):
                        if c in med.index:
                            z_list.append(
                                compute_robust_z(
                                    float(v),
                                    float(med.loc[c]),
                                    float(q25.loc[c]),
                                    float(q75.loc[c]),
                                )
                            )
                        else:
                            z_list.append(float("nan"))
                    z_cap = np.asarray(z_list, dtype=float)
                    if np.any(np.isfinite(z_cap)):
                        cap_min_z = float(np.nanmin(z_cap))
                        onset_i = find_onset(z_cap, thresh=3.0, min_run=3, direction="neg")
                        if onset_i is not None:
                            cap_onset_cycle = int(cycles_arr[onset_i])

            # 2.5) z-score timeline (시각화)
            if (z_dcr is not None) or (z_cap is not None):
                fig_z = go.Figure()
                if z_dcr is not None:
                    fig_z.add_trace(go.Scatter(x=cycles_arr, y=z_dcr, mode="lines", name="Robust z(DCR)"))
                if z_cap is not None:
                    fig_z.add_trace(go.Scatter(x=cycles_arr, y=z_cap, mode="lines", name="Robust z(Capacity%)"))

                fig_z.add_hline(y=3.0, line_width=1, line_dash="dash")
                fig_z.add_hline(y=-3.0, line_width=1, line_dash="dash")

                if dcr_onset_cycle is not None:
                    fig_z.add_vline(x=float(dcr_onset_cycle), line_width=1, line_dash="dot")
                if cap_onset_cycle is not None:
                    fig_z.add_vline(x=float(cap_onset_cycle), line_width=1, line_dash="dot")

                fig_z.update_layout(
                    height=260,
                    margin=dict(l=30, r=10, t=10, b=30),
                    xaxis_title="Cycle",
                    yaxis_title="Robust z",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                )
                st.plotly_chart(fig_z, use_container_width=True)

            # 3) 간단 판정 로직 (데모용)
            issues = []
            if np.isfinite(dcr_max_z) and dcr_max_z >= 4.0:
                issues.append(("Fault-like anomaly (DCR spike)", dcr_onset_cycle, dcr_max_z))
            if np.isfinite(cap_min_z) and cap_min_z <= -3.5:
                issues.append(("Accelerated degradation (Capacity drop)", cap_onset_cycle, cap_min_z))

            # 4) 결과 표시
            if len(issues) == 0:
                st.success("현재 선택된 배터리는 reference cohort의 기대 범위 내에서 큰 이탈이 관측되지 않았습니다.")
            else:
                for name, onset_cycle, score in issues:
                    sev = "HIGH" if (np.isfinite(score) and abs(score) >= 6.0) else "MED"
                    onset_txt = f"cycle {onset_cycle}" if onset_cycle is not None else "(onset 미확정)"
                    st.warning(
                        f"**{name}**  · severity: **{sev}**  · onset: **{onset_txt}**  · robust z: **{score:.2f}**"
                    )

                # 5) '왜?' driver 연결 (onset이 있으면 onset cycle, 아니면 현재 cycle)
                ref_cycle = int(current_cycle)
                if dcr_onset_cycle is not None:
                    ref_cycle = int(dcr_onset_cycle)
                elif cap_onset_cycle is not None:
                    ref_cycle = int(cap_onset_cycle)

                driver_candidates = [
                    "thermal_stress",
                    "temperature_mean",
                    "temp_rise_cycle",
                    "eff_c_rate",
                    "current_max",
                    "current_min",
                    "voltage_min",
                    "dvdt_max_abs",
                    "dTdt_max",
                ]
                drivers = top_driver_explanations(df_b, df_feat, cohort_key_mon, ref_cycle, driver_candidates, topk=3)

                if len(drivers) > 0:
                    st.markdown(f"**Potential drivers around cycle {ref_cycle} (cohort 대비 z-score)**")
                    for d in drivers:
                        st.markdown(
                            f"- **{d['tag']}** · {d['feature']}={d['value']:.4g}, z={d['z']:.2f}  "
                            + (f"→ _{d['action']}_" if d.get("action") else "")
                        )
                else:
                    st.caption("driver 후보 피처가 부족하거나 cohort 분포가 충분하지 않아 Top driver를 계산하지 못했습니다.")

            st.caption(
                "※ 위 판정/설명은 데모용(rule + robust z-score)이며, 실제 운영에서는 조건(cohort)·센서 품질·물리 제약을 함께 고려해 임계치/로직을 튜닝합니다."
            )

with st.expander("🆚 Compare batteries (Geotab-style)", expanded=False):
    df_feat = load_cycle_features()
    if df_feat is None:
        st.info("analysis/nasa_features_rul.csv 를 찾지 못해 비교 그래프를 표시할 수 없습니다.")
    else:
        compare_bids = st.multiselect(
            "비교할 배터리 (NASA test cells)",
            options=all_bids,
            default=[selected_bid],
        )

        metric_options = [
            ("SoH (%)", ("soh", "pct")),
            ("Capacity (% of initial)", ("capacity_mean", "cap_pct")),
            ("Impedance sum (Ω)", ("impedance_sum", "raw")),
            ("DCR (Ω)", ("dcr", "raw")),
            ("Thermal stress", ("thermal_stress", "raw")),
            ("Temp mean (°C)", ("temperature_mean", "raw")),
            ("LLI", ("lli", "raw")),
            ("LAM", ("lam", "raw")),
        ]
        metric_label = st.selectbox("비교 지표", [m[0] for m in metric_options], index=0)
        metric_key, metric_mode = dict(metric_options)[metric_label]

        show_band = st.checkbox("기대 범위(expected band: 중앙값/사분위) 함께 보기", value=True)
        cohort_label = st.selectbox("Reference cohort (expected band)", [c[0] for c in COHORT_OPTIONS], index=0)
        cohort_key = dict(COHORT_OPTIONS)[cohort_label]

        fig_cmp = go.Figure()

        # fleet band (median + IQR) on cycle index
        if show_band and metric_key in df_feat.columns:
            tmp = apply_cohort_filter(df_feat, cohort_key)[["battery_id", "cycle", metric_key]].dropna()
            if not tmp.empty:
                g = tmp.groupby("cycle")[metric_key]
                med = g.median()
                q25 = g.quantile(0.25)
                q75 = g.quantile(0.75)

                y_med = med.values.astype(float)
                y_q25 = q25.values.astype(float)
                y_q75 = q75.values.astype(float)
                x_c = med.index.values.astype(float)

                # normalization for SoH (pct) and capacity pct
                if metric_key == "soh":
                    y_med, y_q25, y_q75 = y_med * 100.0, y_q25 * 100.0, y_q75 * 100.0

                fig_cmp.add_trace(
                    go.Scatter(
                        x=x_c,
                        y=y_q75,
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                        name="q75",
                    )
                )
                fig_cmp.add_trace(
                    go.Scatter(
                        x=x_c,
                        y=y_q25,
                        mode="lines",
                        fill="tonexty",
                        line=dict(width=0),
                        opacity=0.25,
                        hoverinfo="skip",
                        name="IQR (cohort)",
                    )
                )
                fig_cmp.add_trace(
                    go.Scatter(
                        x=x_c,
                        y=y_med,
                        mode="lines",
                        name="Median (cohort)",
                        line=dict(dash="dash"),
                    )
                )

        # per-battery lines
        for bid_ in compare_bids:
            df_one = df_feat[df_feat["battery_id"] == str(bid_)].sort_values("cycle")
            if df_one.empty or metric_key not in df_one.columns:
                continue
            x = df_one["cycle"].astype(float).values
            y = df_one[metric_key].astype(float).values

            # display transforms
            if metric_key == "soh":
                y = y * 100.0
            elif metric_mode == "cap_pct" and metric_key == "capacity_mean":
                try:
                    cap0 = float(df_one.iloc[0]["capacity_mean"])
                    if cap0 > 0:
                        y = (df_one["capacity_mean"].astype(float).values / cap0) * 100.0
                except Exception:
                    pass

            fig_cmp.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines",
                    name=str(bid_),
                    line=dict(width=3 if str(bid_) == str(selected_bid) else 2),
                )
            )

        fig_cmp.add_vline(x=float(current_cycle), line_width=1, line_dash="dash", line_color="red")

        fig_cmp.update_layout(
            height=420,
            margin=dict(l=30, r=10, t=30, b=30),
            xaxis_title="Cycle",
            yaxis_title=metric_label,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)
with st.expander("🧠 Explainability (Global feature importance)", expanded=False):
    if shap_names is None or shap_vals is None or len(shap_names) == 0:
        st.info("SHAP 전역 중요도 파일을 찾지 못했습니다. (shap_outputs/*.json)")
    else:
        try:
            df_shap = pd.DataFrame({"feature": list(shap_names), "importance": list(shap_vals)})
            df_shap = df_shap.dropna()
            df_shap = df_shap.sort_values("importance", ascending=False)
            top_k = st.slider("Top-K features", min_value=5, max_value=min(30, len(df_shap)), value=min(12, len(df_shap)))
            df_top = df_shap.head(int(top_k)).iloc[::-1]

            fig_shap = go.Figure(
                data=[
                    go.Bar(
                        x=df_top["importance"].astype(float).values,
                        y=df_top["feature"].astype(str).values,
                        orientation="h",
                        name="global importance",
                    )
                ]
            )
            fig_shap.update_layout(
                height=380,
                margin=dict(l=30, r=10, t=20, b=30),
                xaxis_title="Importance",
                yaxis_title="Feature",
            )
            st.plotly_chart(fig_shap, use_container_width=True)

            st.caption("※ 이 중요도는 'cycle별'이 아니라 전체 데이터/시퀀스를 통틀어 계산된 전역(global) 중요도입니다.")
        except Exception as e:
            st.error(f"SHAP 중요도 렌더링 중 오류: {e}")

# =================================================
# BOTTOM-LEFT: Usage & Scenario input
# =================================================
with bottom_left:
    st.markdown("### Usage & Scenario input")
    st.caption("※ 슬라이더는 현재 사이클(수동 모드)에서만 조정하는 용도입니다")

    cycles_per_day_sb = st.number_input(
        "평균 하루 주행 사이클 수",
        min_value=0.1,
        max_value=5.0,
        value=st.session_state.get("cycles_per_day", DEFAULT_CYCLES_PER_DAY),
        step=0.1,
        key="cycles_per_day",
    )

    if FEATURE_STATS is None:
        st.warning("feature_rul_stats.json을 찾을 수 없습니다. 먼저 export 스크립트를 실행해 주세요.")
    else:
        # 🔒 자동 재생 중에는 시나리오 슬라이더/계산을 건드리지 않음
        if st.session_state.get("auto_play", False):
            st.info(
                "사이클 자동 재생 중에는 Scenario Builder가 정지됩니다.\n\n"
            )
        elif math.isnan(current_pred_rul):
            st.info("현재 사이클은 예측 구간이 아닙니다. (예측 없음)\n\n왼쪽 슬라이더를 r_ratio 이후 구간으로 옮기면 Scenario 계산이 활성화됩니다.")
        else:
            st.markdown("#### Baseline vs Scenario (열화 조건)")
            st.caption(
                "※ 현재 배터리/사이클에서 Baseline(A)을 기준으로, "
                "Scenario(B) 값을 바꾸면서 잔여수명(RUL)이 어떻게 달라지는지 봅니다."
            )

            cycle_feat_df = load_cycle_features()
            current_feat_row = None
            if cycle_feat_df is not None:
                mask = (cycle_feat_df["battery_id"] == str(selected_bid)) & (
                    cycle_feat_df["cycle"] == int(current_cycle)
                )
                sub = cycle_feat_df.loc[mask]
                if not sub.empty:
                    current_feat_row = sub.iloc[0]

            # 🔹 현재 배터리/사이클 기준 Scenario 컨텍스트 생성
            scenario_ctx = ScenarioContext(selected_bid, current_cycle)

            baseline_vals: Dict[str, float] = {}
            scenario_vals: Dict[str, float] = {}

            feat_items = list(SCENARIO_FEATURES.items())

            # 피처들을 2개씩 묶어서 한 줄에 배치
            for i in range(0, len(feat_items), 2):
                col_left, col_right = st.columns(2)

                for col, (feat_key, meta) in zip(
                    (col_left, col_right),
                    feat_items[i : i + 2],
                ):
                    stats = FEATURE_STATS.get(feat_key, {})

                    # 👉 현재 배터리/사이클 피처값을 Baseline 기본값으로 사용
                    if current_feat_row is not None and feat_key in current_feat_row:
                        base_default = float(current_feat_row[feat_key])
                    else:
                        if "mean" in stats:
                            base_default = float(stats["mean"])
                        else:
                            base_default = 0.5 * (
                                meta["fallback_min"] + meta["fallback_max"]
                            )

                    # ❌ 기존: q10~q90로 클램핑하던 부분 제거
                    baseline_vals[feat_key] = base_default

                    # 🔹 ScenarioContext에 Baseline 등록 + 현재 Scenario 기본값 얻기
                    scen_default = scenario_ctx.register_baseline(
                        feat_key, base_default
                    )

                    # --- 슬라이더 범위 계산 ---
                    q10 = stats.get("q10", None)
                    q90 = stats.get("q90", None)

                    if feat_key in {"dcr", "impedance_sum"}:
                        # 직류저항 / 임피던스 합: 아래로 많이 줄여볼 수 있게 0 ~ 2×현재값 정도로
                        slider_min = 0.0
                        slider_max = max(meta["fallback_max"], base_default * 2.0)
                    elif feat_key == "regen_strength":
                        # 누적 용량 손실: 0에서 시작해서 현재값보다 꽤 더 크게도 가정 가능
                        slider_min = 0.0
                        slider_max = max(meta["fallback_max"], base_default + 0.5)
                    else:
                        # 나머지 피처는 q10~q90를 기본으로 쓰되, 양쪽으로 더 넓게 벌려줌
                        if (q10 is not None) and (q90 is not None) and (q90 > q10):
                            lo_raw = float(q10)
                            hi_raw = float(q90)
                        else:
                            lo_raw = float(meta["fallback_min"])
                            hi_raw = float(meta["fallback_max"])

                        span = hi_raw - lo_raw
                        if span <= 0:
                            slider_min = meta["fallback_min"]
                            slider_max = meta["fallback_max"]
                        else:
                            # q10~q90 범위를 기준으로 ±50% 확장
                            slider_min = min(base_default, lo_raw - 0.5 * span)
                            slider_max = max(base_default, hi_raw + 0.5 * span)

                    # 범위가 너무 좁으면 최소한 ±1 정도는 확보
                    if slider_max - slider_min < 1e-6:
                        slider_min = base_default - 1.0
                        slider_max = base_default + 1.0

                    # Baseline 값이 범위 안에 확실히 들어가도록 한 번 더 정리
                    if base_default < slider_min:
                        slider_min = base_default
                    if base_default > slider_max:
                        slider_max = base_default

                    with col:
                        # 피처 이름
                        st.markdown(f"**{meta['label']}**")

                        # Baseline (A) 박스
                        st.markdown(
                            "<div style='font-size:11px; color:#888;'>Baseline (A)</div>",
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f"<div style='padding:4px 6px; background-color:#f2f2f2;"
                            f" border-radius:4px; font-size:12px; color:#555;'>{base_default:.4g}</div>",
                            unsafe_allow_html=True,
                        )

                        # Scenario (B) 슬라이더
                        st.markdown(
                            "<div style='font-size:11px; color:#c62828;'>Scenario (B)</div>",
                            unsafe_allow_html=True,
                        )

                        # 온도 2개는 고정 (환경 조건) – 누적 용량 손실은 그대로 슬라이더 허용
                        is_locked = feat_key in {"ambient_temp_c", "temp_rise_cycle"}

                        if is_locked:
                            st.markdown(
                                f"<div style='padding:4px 6px; background-color:#fafafa;"
                                f" border-radius:4px; font-size:12px; color:#999;'>{base_default:.4g} (고정)</div>",
                                unsafe_allow_html=True,
                            )
                            scenario_vals[feat_key] = base_default
                        else:
                            scen_val = st.slider(
                                "",
                                min_value=float(slider_min),
                                max_value=float(slider_max),
                                value=float(scen_default),
                                step=meta["step"],
                                key=f"{feat_key}_scenario",
                            )
                            scenario_ctx.update_from_slider(feat_key, scen_val)
                            scenario_vals[feat_key] = float(scen_val)

                        # 🔹 slope 기반 RUL 민감도 설명 (전부 "RUL 증가" 방향 기준으로 표현)
                        slope = float(get_stage_slope(feat_key, life_stage))  # stage-conditioned slope
                        if feat_key in {"ambient_temp_c", "temp_rise_cycle"}:
                            guide = (
                                "NASA 실험에서 상온/고온 로그는 수명 초반, "
                                "저온(4℃)·비정상 용량 로그는 수명 말기에 몰려 있어 "
                                "온도와 수명 단계가 섞인(confounding) 편향이 있습니다. "
                                "그 결과 단순 통계로 보면 RUL과 비직관적인 양의 상관을 보이므로, "
                                "여기서는 온도를 '조절 변수'가 아니라 고정된 조건 피처로만 사용합니다."
                            )
                        elif abs(slope) < 1e-6:
                            guide = "이 피처는 단순 선형 기준에서 RUL과 상관성이 거의 없습니다."
                        elif slope > 0:
                            guide = (
                                f"이 값을 늘리면 RUL이 늘어나는 경향이 있습니다 "
                                f"(선형 근사 기준 +{slope:.1f} cycles/단위)."
                            )
                        else:
                            guide = (
                                f"이 값을 줄이면 RUL이 늘어나는 경향이 있습니다 "
                                f"(값을 1 단위 줄일 때 평균 {abs(slope):.1f} cycles 정도 "
                                f"RUL이 길어지는 방향, 선형 근사 기준)."
                            )

                        st.markdown(
                            f"<div style='font-size:12px; color:#555;'>※ {guide}</div>",
                            unsafe_allow_html=True,
                        )

            # 시나리오 계산에서 쓸 입력들을 세션에 저장
            st.session_state["scenario_inputs"] = {
                "baseline": baseline_vals,
                "scenario": scenario_vals,
                "cycles_per_day": cycles_per_day_sb,
                "current_pred_rul": current_pred_rul,
                "current_true_rul": current_true_rul,
                "initial_rul": initial_rul,
                "current_cycle": current_cycle,
                "life_stage": life_stage,
                "battery_id": selected_bid,
            }


# =================================================
# BOTTOM-RIGHT: Scenario 계산 + 결과
# =================================================
with bottom_right:
    st.markdown("### Scenario results")

    scenario_inputs = st.session_state.get("scenario_inputs", None)

    # 🔁 Scenario 리셋 버튼
    reset_clicked = st.button(
        "Scenario 리셋",
        help="슬라이더와 Scenario results를 현재 Baseline 값으로 되돌립니다.",
        disabled=(scenario_inputs is None),
    )

    if reset_clicked and scenario_inputs is not None:
        # 현재 컨텍스트 기준으로 Scenario 를 전부 Baseline으로 되돌림
        baseline_for_reset = scenario_inputs["baseline"]
        ctx_for_reset = ScenarioContext(
            scenario_inputs["battery_id"], scenario_inputs["current_cycle"]
        )
        ctx_for_reset.reset_to_baseline(baseline_for_reset)

        if hasattr(st, "rerun"):
            st.rerun()
        elif hasattr(st, "experimental_rerun"):
            st.experimental_rerun()

    # 이 아래부터는 Scenario 결과 계산 로직
    scenario_inputs = st.session_state.get("scenario_inputs", None)

    # 🔒 auto_play 중에는 Scenario results 자체를 계산하지 않음
    if st.session_state.get("auto_play", False):
        st.info(
            "사이클 자동 재생 중에는 Scenario results가 갱신되지 않습니다. "
            "자동 재생을 끄고 시나리오를 조정해 주세요."
        )
    elif scenario_inputs is None or FEATURE_STATS is None:
        st.info("왼쪽에서 baseline / scenario 조건을 먼저 설정해 주세요.")
    else:
        baseline = scenario_inputs["baseline"]          # dict: feat_key -> baseline 값
        scenario = scenario_inputs["scenario"]          # dict: feat_key -> scenario 값
        cycles_per_day_sb = scenario_inputs["cycles_per_day"]

        current_pred_rul_s = float(scenario_inputs["current_pred_rul"])
        current_true_rul_s = float(scenario_inputs["current_true_rul"])

        if math.isnan(current_pred_rul_s):
            st.info(
                """현재 사이클은 예측 구간이 아니라 Scenario 결과를 계산할 수 없습니다. (예측 없음)

r_ratio 이후 구간으로 이동해 주세요."""
            )
            st.stop()
        initial_rul_s = float(scenario_inputs["initial_rul"])
        current_cycle_s = int(scenario_inputs["current_cycle"])
        life_stage_s = str(scenario_inputs.get("life_stage", "mid"))
        selected_bid_s = scenario_inputs["battery_id"]

        # 1) 시나리오 결과 계산 (선형 민감도 기반 RUL 보정)
        delta_rul = 0.0
        per_feat_details = []

        for feat_key in SCENARIO_FEATURES.keys():
            base_v = float(baseline[feat_key])
            scen_v = float(scenario[feat_key])
            delta_v = scen_v - base_v

            stats = FEATURE_STATS.get(feat_key, {})
            slope = float(get_stage_slope(feat_key, life_stage_s))  # stage-conditioned slope (early/mid/late)
            contrib = slope * delta_v                             # ΔRUL_j = slope_j * Δx_j
            delta_rul += contrib

            per_feat_details.append(
                {
                    "feature": feat_key,
                    "base": base_v,
                    "scenario": scen_v,
                    "delta_val": delta_v,
                    "slope_rul_per_unit": slope,
                    "delta_rul": contrib,
                }
            )

        # 최종 Scenario RUL (현재 예측 RUL + ΣΔRUL), 음수는 0으로 클램프
        scen_rul = max(0.0, current_pred_rul_s + delta_rul)

        # 남은 일수 환산
        remaining_days_base = current_pred_rul_s / max(cycles_per_day_sb, 1e-6)
        remaining_days_scen = scen_rul / max(cycles_per_day_sb, 1e-6)
        diff_cycles = scen_rul - current_pred_rul_s

        # 실제로 바뀐 피처가 있는지 여부 (delta_val ≈ 0이면 "안 바뀐 것" 취급)
        any_changed = any(abs(d["delta_val"]) >= 1e-6 for d in per_feat_details)

        # 2) RUL 요약 카드 (Baseline / Scenario / ΔRUL)
        base_bg = "#f2f2f2"
        base_color = "#555555"
        scen_bg = "#e8f1ff"
        scen_color = "#1f77b4"

        if diff_cycles >= 0:
            delta_bg = "#e8f5e9"
            delta_color = "#2e7d32"
            delta_label = "RUL 증가 (Δ)"
        else:
            delta_bg = "#ffebee"
            delta_color = "#c62828"
            delta_label = "RUL 감소 (Δ)"

        c1, c2, c3 = st.columns(3)

        # Baseline RUL은 항상 표시
        with c1:
            st.markdown(
                f"""
                <div style='background-color:{base_bg}; padding:8px 12px; border-radius:8px; text-align:center;'>
                  <div style='font-size:11px; color:#777;'>Baseline RUL</div>
                  <div style='font-size:20px; font-weight:600; color:{base_color};'>
                    {current_pred_rul_s:.1f} cycles
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if any_changed:
            # Scenario RUL / ΔRUL: 실제로 피처를 바꿨을 때만 숫자 표시
            with c2:
                st.markdown(
                    f"""
                    <div style='background-color:{scen_bg}; padding:8px 12px; border-radius:8px; text-align:center;'>
                      <div style='font-size:11px; color:#1f4b8f;'>Scenario RUL</div>
                      <div style='font-size:20px; font-weight:600; color:{scen_color};'>
                        {scen_rul:.1f} cycles
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c3:
                st.markdown(
                    f"""
                    <div style='background-color:{delta_bg}; padding:8px 12px; border-radius:8px; text-align:center;'>
                      <div style='font-size:11px; color:{delta_color};'>{delta_label}</div>
                      <div style='font-size:20px; font-weight:600; color:{delta_color};'>
                        {diff_cycles:+.1f} cycles
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            # 아직 피처를 건드리지 않은 상태: 안내 문구만 표시
            with c2:
                st.markdown(
                    """
                    <div style='background-color:#f8f9fb; padding:8px 12px; border-radius:8px; text-align:center;'>
                      <div style='font-size:11px; color:#1f4b8f;'>Scenario RUL</div>
                      <div style='font-size:12px; color:#999; margin-top:4px;'>
                        Scenario(B)를 변경하면 표시됩니다.
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            with c3:
                st.markdown(
                    """
                    <div style='background-color:#f8f9fb; padding:8px 12px; border-radius:8px; text-align:center;'>
                      <div style='font-size:11px; color:#777;'>ΔRUL</div>
                      <div style='font-size:12px; color:#999; margin-top:4px;'>
                        변경된 피처가 없습니다.
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # 3) 피처별 기여: "바뀐 피처와 RUL 영향 (간단 요약)"
        with st.expander("바뀐 피처와 RUL 영향 (간단 요약)", expanded=True):
            if not any_changed:
                st.caption("이번 시나리오에서 바뀐 피처가 없습니다.")
            else:
                for d in per_feat_details:
                    if abs(d["delta_val"]) < 1e-6:
                        continue  # 안 바뀐 피처는 스킵
                    feat_key = d["feature"]
                    meta = SCENARIO_FEATURES.get(feat_key, {})
                    label = meta.get("label", feat_key)
                    arrow = "↑" if d["delta_val"] > 0 else "↓"

                    st.markdown(
                        f"- **{label}** {arrow} "
                        f"({d['base']:.4g} → {d['scenario']:.4g}) → "
                        f"RUL {d['delta_rul']:+.1f} cycles"
                    )

        st.markdown("---")

        # 4) 지금 상태를 시나리오 테이블에 저장
        add_clicked = st.button("현재 설정을 시나리오 테이블에 추가")

        if add_clicked:
            scenario_row = {
                "battery_id": selected_bid_s,
                "cycle": current_cycle_s,
                "rul_true": current_true_rul_s,
                "rul_model_base": current_pred_rul_s,
                "rul_model_scenario": scen_rul,
                "cycles_per_day": cycles_per_day_sb,
                "remaining_days_base": remaining_days_base,
                "remaining_days_scenario": remaining_days_scen,
            }
            for d in per_feat_details:
                scenario_row[f"{d['feature']}_delta"] = d["delta_val"]

            st.session_state["scenarios"].append(scenario_row)
            st.success(
                f"시나리오 추가 완료: {current_pred_rul_s:.1f} → {scen_rul:.1f} cycles "
                f"({diff_cycles:+.1f} cycles)"
            )

        # 5) 누적 시나리오 테이블
        if st.session_state["scenarios"]:
            st.markdown("#### 누적 시나리오 테이블")

            df_scen = pd.DataFrame(st.session_state["scenarios"])
            st.dataframe(df_scen, use_container_width=True, height=260)

            csv_bytes = df_scen.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download scenarios as CSV",
                data=csv_bytes,
                file_name="rul_scenarios.csv",
                mime="text/csv",
            )

# =================================================
# Auto-play 루프 (non-blocking preferred)
# =================================================
if st.session_state.get("auto_play", False):
    # Prefer st_autorefresh (non-blocking) if installed
    if st_autorefresh is not None:
        tick = st_autorefresh(interval=int(AUTO_PLAY_DELAY_SEC * 1000), key="autoplay_refresh")
        last_tick = st.session_state.get("_autoplay_last_tick", None)

        # tick이 바뀐 시점에만 1 step 진행
        if tick != last_tick:
            st.session_state["_autoplay_last_tick"] = tick
            cur = int(st.session_state.get("play_cycle", current_cycle))
            if cur < max_cycle:
                st.session_state["play_cycle"] = min(max_cycle, cur + AUTO_PLAY_STEP)
            else:
                st.session_state["auto_play"] = False
    else:
        # Fallback: blocking sleep + rerun (older Streamlit)
        cur = st.session_state.get("play_cycle", current_cycle)
        if cur < max_cycle:
            next_cycle = min(max_cycle, cur + AUTO_PLAY_STEP)
            st.session_state["play_cycle"] = next_cycle

            time.sleep(AUTO_PLAY_DELAY_SEC)

            if hasattr(st, "rerun"):
                st.rerun()
            elif hasattr(st, "experimental_rerun"):
                st.experimental_rerun()

st.markdown("---")

st.markdown("---")

st.markdown(
"""

<div style="font-size:13px; line-height:1.6; color:#222; max-width:820px; margin:0 auto;">

이 대시보드는 NASA 실험실에서 측정한 셀 사이클 데이터를 기반으로 학습한
잔여수명(RUL) 예측 모델의 동작을 보여줍니다.

<b>상단 Battery remaining life trajectory (RUL) </b><br/>

* 왼쪽에서 초기 적응 구간(r_ratio), 배터리, 자동 재생 유무 등을 선택하면
  prefix 이후 구간에서의 예측, 불확실성 범위, 오차 지표를 함께 확인할 수 있습니다.<br/>
* 오른쪽 그래프는 사이클에 따른 실제 RUL과 모델이 예측한 RUL 궤적을 함께 보여줍니다.<br/><br/>

<b>하단 Scenario Builder (what-if 시뮬레이션)</b><br/>

* 선택한 배터리/사이클에 대해 nasa_features_rul.csv에 들어 있는 실제 피처 값을 Baseline (A)으로 사용합니다.<br/>
* 사용자가 슬라이더로 바꾼 값은 Scenario (B)로 사용합니다.<br/>
* 즉, “지금 이 셀/사이클 상태에서 특정 피처를 이렇게 바꾸면 RUL이 대략 얼마나 달라질까?”를
  보는 가벼운 what-if 도구입니다.<br/><br/>

<b>시뮬레이션(RUL 보정) 개념</b><br/>

* feature_rul_stats.json에는 각 피처 값이 1 단위 변할 때 RUL이 평균 몇 사이클 변하는지에 대한
  기울기(민감도)가 저장되어 있습니다. (NASA 전체 데이터 기준)<br/>
* 현재 선택한 배터리/사이클의 실제 피처 값(Baseline)과 슬라이더로 조정한 값(Scenario)의 차이를 이용해
  피처별 RUL 변화량을 선형으로 근사한 뒤 모두 더해 Scenario RUL을 계산합니다.<br/>
* 즉, 모델이 그려준 RUL 곡선을 기준선으로 두고, 피처 변화에 따른 RUL 보정을
  선형 민감도로 더해주는 post-hoc what-if 시뮬레이션입니다.<br/><br/>

본 대시보드의 RUL 추정치는 NASA PCoE Li-ion #25–#56 실험 데이터에서 학습한 통계 모델 결과로,
데이터셋 편향과 confounding의 영향을 일부 포함합니다. 따라서 실제 운용 수명의 절대값이라기보다는
상태 비교와 열화 추세 파악을 위한 참고 지표로 사용하는 것을 권장합니다.

</div>
    """,
    unsafe_allow_html=True,
)