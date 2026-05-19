import json
import math
import functools
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

st.set_page_config(
    page_title="Battery RUL Meta-Learning Dashboard (Real-time Inference)",
    layout="wide",
)

# ---- Optional autorefresh (used only when autoplay is ON) ----
try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
except Exception:
    st_autorefresh = None  # fallback

# ---- Cache heavy data loads to avoid UI freeze/flicker on rerun ----
@functools.lru_cache(maxsize=16)
def _load_csv_cached(path: str, **kwargs) -> pd.DataFrame:
    import pandas as _pd
    return _pd.read_csv(path, **kwargs)


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

BMAML_DIR = FILE_DIR / "dashboard_export_v2" / "bmaml_v2"
CKPT_DEFAULT = FILE_DIR / "core_checkpoints" / "nasa_bmaml_best_re.pt"
SHAP_JSON = FILE_DIR / "shap_outputs" / "bmaml_shap_seq_feature_importance.json"

FEATURE_STATS_JSON_PATH = FILE_DIR / "analysis" / "feature_rul_stats.json"
FEATURE_STATS_CSV_PATH = FILE_DIR / "analysis" / "feature_rul_stats.csv"

# Prefer CSV if available (easier to inspect/version)
FEATURE_STATS_PATH = FEATURE_STATS_CSV_PATH if FEATURE_STATS_CSV_PATH.exists() else FEATURE_STATS_JSON_PATH
# --- Cycle feature table (prefers enriched CSV) ---
def _first_existing(paths):
    for p in paths:
        try:
            if p is not None and Path(p).exists():
                return Path(p)
        except Exception:
            continue
    return None

SCRIPT_DIR = FILE_DIR
CWD = Path.cwd()

NASA_FEATURES_PATH_CANDIDATES = [
    # Enriched (preferred)
    SCRIPT_DIR / "analysis" / "battery_training_data_cleaned_final_causal_with_true_crate_v2_expcond_temp0.csv",
    SCRIPT_DIR / "battery_training_data_cleaned_final_causal_with_true_crate_v2_expcond_temp0.csv",
    CWD / "analysis" / "battery_training_data_cleaned_final_causal_with_true_crate_v2_expcond_temp0.csv",
    CWD / "battery_training_data_cleaned_final_causal_with_true_crate_v2_expcond_temp0.csv",
    # Fallback (may miss engineered columns)
    SCRIPT_DIR / "analysis" / "nasa_features_rul.csv",
    CWD / "analysis" / "nasa_features_rul.csv",
    SCRIPT_DIR / "analysis" / "nasa_features_rul_precomputed.csv",
    CWD / "analysis" / "nasa_features_rul_precomputed.csv",
]

def _pick_cycle_features_path(cands):
    # Add rglob hits (common when running from a different working dir)
    for base in [SCRIPT_DIR, CWD]:
        try:
            for p in base.rglob("battery_training_data_cleaned_final_causal_with_true_crate_v2_expcond_temp0.csv"):
                if p not in cands:
                    cands.insert(0, p)
        except Exception:
            pass

    existing = [p for p in cands if p is not None and Path(p).exists()]

    if not existing:
        return None

    want = {"discharge_time_sec", "c_ref_ahr", "c_rate_peak", "c_rate_mean", "eff_c_rate_C"}

    def score(p):
        try:
            cols = set(_load_csv_cached(p, nrows=5).columns)
            return sum((w in cols) for w in want)
        except Exception:
            return -1

    best = max(existing, key=score)
    if score(best) > 0:
        return best

    return _first_existing(existing)

NASA_FEATURES_PATH = _pick_cycle_features_path(NASA_FEATURES_PATH_CANDIDATES)


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

@st.cache_data(show_spinner=False)
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


@st.cache_data(show_spinner=False)
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
@st.cache_data(show_spinner=False)
def load_cycle_features():
    """사이클별 피처 테이블 로드 (battery_id, cycle로 컬럼 통일)."""
    if NASA_FEATURES_PATH is None or (not NASA_FEATURES_PATH.exists()):
        return None
    df = _load_csv_cached(NASA_FEATURES_PATH)

    # battery_id
    if "battery_id" in df.columns:
        df["battery_id"] = df["battery_id"].astype(str)
    elif "battery" in df.columns:
        df["battery_id"] = df["battery"].astype(str)
    else:
        return None

    # cycle
    if "cycle" in df.columns:
        df["cycle"] = df["cycle"].astype(int)
    elif "cycle_num" in df.columns:
        df.rename(columns={"cycle_num": "cycle"}, inplace=True)
        df["cycle"] = df["cycle"].astype(int)
    else:
        return None

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
        df = _load_csv_cached(path)
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
        # cohort 분산(IQR)이 거의 0이면 z-score가 비정상적으로 커질 수 있어 스킵
        if float(q75 - q25) < 1e-6:
            continue
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

    if float(r_ratio) <= 0.0:
        st.error("r_ratio=0.00 은 초기 적응 데이터가 0개이므로 실시간 재추론을 수행할 수 없습니다. 0.05 이상으로 설정해 주세요.")
        st.stop()
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
def _metric_card(label_html: str, value_html: str):
    # Compact KPI card for the overview board
    st.markdown(
        f"""
        <div style="border:1px solid #e6e6e6; border-radius:12px; padding:8px 10px; min-height:70px;">
          <div style="font-size:11px; color:#666; line-height:1.15; white-space:normal;">{label_html}</div>
          <div style="font-size:20px; font-weight:700; margin-top:6px; white-space:normal;">{value_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


MODEL_TAG = "Fleet Battery Health APM / Analytics — Few-shot meta-learned RUL predictor (Transformer backbone)"
st.title("nasa 리튬 이온 배터리 실험 데이터를 기반으로 한 잔여수명(RUL) 예측 모델의 개발")
st.caption("이 모델 개발의 목적은 배터리별 실험 조건과 셀 길이의 차이를 극복한 범용적인 모델 개발을 위해 BMAML-SVGD 기반 베이지안 메타러닝(ceemdan 분해와 transformer 백본 위에 얹은)을 통해 처음 본 배터리에서 최소한의 초기 데이터로 열화패턴(곡선)을 학습하여 잔여수명 곡선을 예측하였다.")
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

# 🔽 r_ratio 선택 옵션 (discrete)
R_RATIO_OPTIONS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

def _snap_ratio(v: float) -> float:
    try:
        v = float(v)
    except Exception:
        v = float(DEFAULT_R_RATIO)
    return min(R_RATIO_OPTIONS, key=lambda x: abs(x - v))

slider_default = _snap_ratio(st.session_state.get("r_ratio", DEFAULT_R_RATIO))

r_ratio = st.sidebar.select_slider(
    "초기 적응 비율 (r_ratio)",
    options=R_RATIO_OPTIONS,
    value=slider_default,
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

# 수동 모드: 현재 사이클 선택
# 🔁 Auto-play 체크박스 (원래 방식)
auto_play = st.sidebar.checkbox("사이클 자동 재생", key="auto_play", value=False)

current_cycle_slider = st.sidebar.slider(
    "현재 사이클 (수동 모드)",
    min_value=min_cycle,
    max_value=max_cycle,
    step=1,
    value=int(st.session_state.get("play_cycle", max(min_cycle, 1))),
    key="current_cycle_slider",
    disabled=st.session_state.get("auto_play", False),
)

# 내부 play_cycle 동기화
if "play_cycle" not in st.session_state:
    st.session_state["play_cycle"] = int(current_cycle_slider)

if st.session_state.get("auto_play", False):
    # Prefer non-blocking refresh if available
    if st_autorefresh is not None:
        st_autorefresh(interval=1200, key="autoplay_refresh")
        st.session_state["play_cycle"] = min(max_cycle, int(st.session_state["play_cycle"]) + 1)
    else:
        # Fallback: blocking sleep + rerun (will show loading overlay)
        time.sleep(1.2)
        st.session_state["play_cycle"] = min(max_cycle, int(st.session_state["play_cycle"]) + 1)
        if hasattr(st, "rerun"):
            st.rerun()
        elif hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
    current_cycle = int(st.session_state["play_cycle"])
else:
    current_cycle = int(current_cycle_slider)
    st.session_state["play_cycle"] = current_cycle

st.sidebar.caption(
    "※ r_ratio 이후 구간(예: 25% 이후)으로 옮기면, "
    "모델이 예측한 RUL 구간부터 바로 볼 수 있습니다."
)
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
# Main content (tabs)
# =================================================
# Page selector (executes only the selected page to avoid heavy reruns/flicker)
page = st.sidebar.radio(
    "탭",
    ["Overview", "Anomaly", "Compare / Fleet", "What-if"],
    index=0,
)
# -----------------------------
# Helpers for Monitoring / Compare
# -----------------------------
def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

def _compute_cap_pct(df_feat: pd.DataFrame) -> pd.DataFrame:
    if df_feat is None or df_feat.empty:
        return df_feat
    if "capacity_mean" not in df_feat.columns:
        return df_feat
    # cap0 per battery
    cap0 = (
        df_feat.sort_values(["battery_id", "cycle"])
        .groupby("battery_id")["capacity_mean"]
        .first()
        .rename("cap0")
    )
    out = df_feat.merge(cap0, left_on="battery_id", right_index=True, how="left")
    out["cap_pct"] = (out["capacity_mean"] / out["cap0"]) * 100.0
    return out

@st.cache_data(show_spinner=False)
def _cycle_band_table(df_feat_all: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Per-cycle median + Q1/Q3 (IQR band) for a feature."""
    if df_feat_all is None or df_feat_all.empty or feature not in df_feat_all.columns:
        return pd.DataFrame(columns=["cycle", "median", "q1", "q3", "iqr"])
    g = df_feat_all.groupby("cycle")[feature]
    band = pd.DataFrame(
        {
            "median": g.median(),
            "q1": g.quantile(0.25),
            "q3": g.quantile(0.75),
        }
    ).reset_index()
    band["iqr"] = (band["q3"] - band["q1"]).replace(0.0, np.nan)
    return band

def _zscore_against_cycle_band(df_b: pd.DataFrame, band: pd.DataFrame, feature: str, z_sign: str = "auto") -> pd.Series:
    """z = (x - median) / IQR. z_sign can be 'auto' or '+', '-' for display only."""
    if df_b is None or df_b.empty or band is None or band.empty:
        return pd.Series(dtype=float)
    m = df_b[["cycle", feature]].merge(band[["cycle", "median", "iqr"]], on="cycle", how="left")
    z = (m[feature] - m["median"]) / m["iqr"]
    return z

def _first_onset_cycle(cycles_arr: np.ndarray, z_arr: np.ndarray, thr: float = 2.0, consec: int = 3) -> int:
    """First cycle where |z| >= thr for consec consecutive points. Returns -1 if none."""
    if cycles_arr.size == 0 or z_arr.size == 0:
        return -1
    mask = np.abs(z_arr) >= thr
    run = 0
    for i, ok in enumerate(mask):
        run = run + 1 if ok else 0
        if run >= consec:
            return int(cycles_arr[i - consec + 1])
    return -1

# Load full cycle-features once (used by Monitoring/Compare)
df_feat_all = load_cycle_features()
if df_feat_all is not None:
    df_feat_all = df_feat_all.copy()
    # normalize types
    if "battery_id" in df_feat_all.columns:
        df_feat_all["battery_id"] = df_feat_all["battery_id"].astype(str)
    if "cycle" in df_feat_all.columns:
        df_feat_all["cycle"] = pd.to_numeric(df_feat_all["cycle"], errors="coerce")
    df_feat_all = _compute_cap_pct(df_feat_all)

# -----------------------------
# OVERVIEW TAB
# -----------------------------
if page == "Overview":
    left, right = st.columns([1.25, 1.75], gap="large")

    # --- Left: key metrics + quick degradation snapshot
    with left:
        st.markdown(f"## Battery {selected_bid}")

        # Risk badge (simple but useful)
        if not math.isnan(current_pred_rul) and initial_rul > 0:
            rul_pct = max(0.0, min(100.0, 100.0 * float(current_pred_rul) / float(initial_rul)))
            if rul_pct >= 60.0:
                risk_lbl = "OK"
                risk_color = "#2e7d32"
            elif rul_pct >= 30.0:
                risk_lbl = "WARN"
                risk_color = "#ff7f0e"
            else:
                risk_lbl = "ALERT"
                risk_color = "#c62828"
        else:
            rul_pct = float("nan")
            risk_lbl = "N/A"
            risk_color = "#777777"

        # Current uncertainty at cursor (±2σ)
        cur_pi = float("nan")
        if in_future_region and fut_cycles.size > 0 and fut_rul_std.size > 0 and not math.isnan(current_pred_rul):
            try:
                idx_pi = int(np.argmin(np.abs(fut_cycles - float(current_cycle))))
                cur_std = float(fut_rul_std[idx_pi])
                if not math.isnan(cur_std):
                    cur_pi = 2.0 * cur_std
            except Exception:
                cur_pi = float("nan")

        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Current cycle", int(current_cycle))
        with m2:
            st.metric("Pred RUL (cycles)", f"{current_pred_rul:.1f}" if not math.isnan(current_pred_rul) else "N/A")
        with m3:
            st.markdown(
                f"""
                <div style="border:1px solid #e6e6e6; border-radius:10px; padding:8px 10px; text-align:center;">
                  <div style="font-size:11px; color:#666;">Risk</div>
                  <div style="font-size:16px; font-weight:700; color:{risk_color};">{risk_lbl}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if not math.isnan(cur_pi):
            st.caption(f"Prediction confidence: ±2σ ≈ {cur_pi:.2f} cycles")
        else:
            st.caption("Prediction confidence: N/A (not in prediction region)")

                # Overview signals are rendered below the RUL trajectory (full-width board)
    with right:
        st.markdown("## RUL trajectory")

        fig = go.Figure()

        # 1) True RUL (offline reference)
        if cycles.size > 0 and all_true.size == cycles.size:
            fig.add_trace(
                go.Scatter(
                    x=cycles,
                    y=all_true,
                    mode="lines",
                    name="RUL (true)",
                    line=dict(color="rgba(0,0,0,0.5)", dash="dash"),
                )
            )

        # 2) Prediction + uncertainty from current cycle onward
        if in_future_region and fut_cycles.size > 0 and fut_rul_pred.size > 0:
            mask_future = fut_cycles >= float(current_cycle)
            fut_x = fut_cycles[mask_future]
            fut_pred_y = fut_rul_pred[mask_future]
            fut_std_y = fut_rul_std[mask_future] if fut_rul_std.size == fut_rul_pred.size else np.full_like(fut_pred_y, np.nan)

            if fut_x.size > 0:
                # Uncertainty band (±2σ)
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
                            name="Uncertainty (±2σ)",
                        )
                    )

                fig.add_trace(
                    go.Scatter(
                        x=fut_x,
                        y=fut_pred_y,
                        mode="lines",
                        name="RUL (pred)",
                        line=dict(color="rgb(214,39,40)"),
                    )
                )

        # Cursor vline
        fig.add_vline(
            x=float(current_cycle),
            line_width=2,
            line_dash="dash",
            line_color="red",
            annotation_text="current cycle",
            annotation_position="top right",
        )

        # EOL reference (true)
        if cycles.size > 0 and all_true.size == cycles.size:
            eol_idxs = np.where(all_true <= 0)[0]
            if eol_idxs.size > 0:
                eol_cycle = float(cycles[eol_idxs[0]])
                fig.add_vline(
                    x=eol_cycle,
                    line_width=2,
                    line_dash="dash",
                    line_color="orange",
                    annotation_text="EOL (true)",
                    annotation_position="top right",
                )

        fig.update_layout(
            xaxis_title="Cycle",
            yaxis_title="RUL (cycles)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=20, t=10, b=30),
            height=520,
        )
        st.plotly_chart(fig, use_container_width=True, key='rul_trajectory')

    # ----- Signals board (below the chart, full width) -----
    
    
    # ----- Signals board (below the chart, full width) -----
    st.markdown("### Overview signals")
    st.caption(f"현재 선택된 사이클에서의 cycle-level 피처 값  |  battery={selected_bid}  cycle={current_cycle}")

    row = getattr(cycle_ctx, "row", None)

    def _num(key, default=np.nan):
        if row is None:
            return default
        if key in row and pd.notna(row[key]):
            try:
                return float(row[key])
            except Exception:
                return default
        return default

    def _str(key, default="—"):
        if row is None:
            return default
        if key in row and pd.notna(row[key]):
            try:
                return str(row[key])
            except Exception:
                return default
        return default

    # -------- base values from DB --------
    capacity_ahr_v = _num("capacity_ahr")
    soh_v = _num("soh")
    capacity_derivative_v = _num("capacity_derivative")

    voltage_measured_mean_v = _num("voltage_measured_mean")
    voltage_min_v = _num("voltage_min")
    current_mean_v = _num("current_mean")
    current_max_v = _num("current_max")
    current_min_v = _num("current_min")

    discharge_time_sec_v = _num("discharge_time_sec")
    eff_c_rate_v = _num("eff_c_rate")

    c_ref_ahr_v = _num("c_ref_ahr")
    c_rate_peak_v = _num("c_rate_peak")
    c_rate_mean_v = _num("c_rate_mean")
    eff_c_rate_C_v = _num("eff_c_rate_C")
    discharge_current_class_v = _str("discharge_current_class", default="—")

    temp_start_v = _num("ambient_temp_c")  # (= Temperature_measured[0] at discharge start, °C) in this DB
    temp_max_v = _num("temperature_measured_max")
    temp_min_v = _num("temperature_min")
    temp_rise_cycle_v = _num("temp_rise_cycle")
    thermal_stress_v = _num("thermal_stress")
    dvdt_max_abs_v = _num("dvdt_max_abs")
    dTdt_max_v = _num("dTdt_max")
    exp_cond_v = _str("experiment_condition", default="")

    dcr_v = _num("dcr")
    dcr_growth_v = _num("dcr_growth")
    lli_v = _num("lli")
    lam_v = _num("lam")

    # -------- derived / fallbacks (keeps UI from showing "—" when DB has enough primitives) --------
    # temp_rise_cycle fallback
    if not np.isfinite(temp_rise_cycle_v) and np.isfinite(temp_max_v) and np.isfinite(temp_start_v):
        temp_rise_cycle_v = temp_max_v - temp_start_v

    # experiment condition fallback (실온/4도)
    if exp_cond_v in ["", "nan", "None"]:
        if np.isfinite(temp_start_v):
            exp_cond_v = "저온 4°C" if temp_start_v <= 10.0 else "실온"
        else:
            exp_cond_v = "—"

    # c_ref_ahr fallback (per-battery early-life median of capacity_ahr)
    if (not np.isfinite(c_ref_ahr_v)) and np.isfinite(capacity_ahr_v):
        try:
            df_all = load_cycle_features()
            if df_all is not None:
                sub_b = df_all[df_all["battery_id"] == str(selected_bid)].sort_values("cycle")
                if "capacity_ahr" in sub_b.columns and len(sub_b) > 0:
                    k_ref = 10
                    c_ref_ahr_v = float(np.nanmedian(sub_b["capacity_ahr"].values[:k_ref]))
        except Exception:
            pass

    # c_rate_* and eff_c_rate_C fallbacks
    if (not np.isfinite(c_rate_peak_v)) and np.isfinite(c_ref_ahr_v) and np.isfinite(current_min_v) and (c_ref_ahr_v > 0):
        c_rate_peak_v = abs(current_min_v) / c_ref_ahr_v
    if (not np.isfinite(c_rate_mean_v)) and np.isfinite(c_ref_ahr_v) and np.isfinite(current_mean_v) and (c_ref_ahr_v > 0):
        c_rate_mean_v = abs(current_mean_v) / c_ref_ahr_v
    if (not np.isfinite(eff_c_rate_C_v)) and np.isfinite(c_ref_ahr_v) and np.isfinite(eff_c_rate_v) and (c_ref_ahr_v > 0):
        eff_c_rate_C_v = eff_c_rate_v / c_ref_ahr_v

    # discharge_current_class fallback (1A/2A/4A)
    if discharge_current_class_v in ["—", "", "nan", "None"]:
        if np.isfinite(current_min_v):
            a = abs(current_min_v)
            if a >= 3.0:
                discharge_current_class_v = "4A"
            elif a >= 1.5:
                discharge_current_class_v = "2A"
            else:
                discharge_current_class_v = "1A"

    # thermal_stress fallback: exp(alpha * (Tmax - Tref))
    if not np.isfinite(thermal_stress_v) and np.isfinite(temp_max_v):
        try:
            alpha = 0.05
            df_all = load_cycle_features()
            if df_all is not None and "temperature_measured_max" in df_all.columns:
                sub_b = df_all[df_all["battery_id"] == str(selected_bid)].sort_values("cycle")
                k_ref = 10
                tref = float(np.nanmedian(sub_b["temperature_measured_max"].values[:k_ref])) if len(sub_b) else float("nan")
                if np.isfinite(tref):
                    thermal_stress_v = float(np.exp(alpha * (temp_max_v - tref)))
        except Exception:
            pass

    # -------- layout --------
    # Capacity
    st.markdown("#### Conditions")
    cnd1, cnd2 = st.columns(2, gap="small")
    with cnd1:
        _metric_card("discharge_current_class<br>(1A/2A/4A)", f"{discharge_current_class_v}")
    with cnd2:
        _metric_card("실험조건 (실온/4도)", exp_cond_v)

    st.markdown("#### Capacity")
    cap1, cap2, cap3 = st.columns(3, gap="small")
    with cap1:
        _metric_card("capacity_ahr<br>(end-of-discharge capacity, Ah)", f"{capacity_ahr_v:.3f} Ah" if np.isfinite(capacity_ahr_v) else "—")
    with cap2:
        _metric_card("soh<br>(capacity(% initial))", f"{soh_v:.3f}" if np.isfinite(soh_v) else "—")
    with cap3:
        _metric_card("capacity_derivative = diff(C)<br>/사이클 간 용량 변화량(ΔCapacity)", f"{capacity_derivative_v:.4f} Ah" if np.isfinite(capacity_derivative_v) else "—")
    st.markdown("#### Load/Voltage")
    lv0, lv1, lv2, lv3, lv4 = st.columns(5, gap="small")
    with lv0:
        _metric_card("discharge_current_class<br>(1A/2A/4A)", f"{discharge_current_class_v}")
    with lv1:
        _metric_card("voltage_measured_mean", f"{voltage_measured_mean_v:.3f} V" if np.isfinite(voltage_measured_mean_v) else "—")
    with lv2:
        _metric_card("voltage_min", f"{voltage_min_v:.3f} V" if np.isfinite(voltage_min_v) else "—")
    with lv3:
        _metric_card("current_mean", f"{current_mean_v:.3f} A" if np.isfinite(current_mean_v) else "—")
    with lv4:
        _metric_card("current_max", f"{current_max_v:.3f} A" if np.isfinite(current_max_v) else "—")

    lv5, lv6, lv7, lv8 = st.columns(4, gap="small")
    with lv5:
        _metric_card("discharge_time_sec", f"{discharge_time_sec_v:.0f} s" if np.isfinite(discharge_time_sec_v) else "—")
    with lv6:
        _metric_card("eff_c_rate<br>(capacity_ahr / (discharge_time_sec/3600) (단위 A))", f"{eff_c_rate_v:.3f} A" if np.isfinite(eff_c_rate_v) else "—")
    with lv7:
        _metric_card("c_ref_ahr<br>(battery별 초기 용량 기준값)", f"{c_ref_ahr_v:.3f} Ah" if np.isfinite(c_ref_ahr_v) else "—")
    with lv8:
        _metric_card("c_rate_peak = abs(current_min)<br>/ c_ref_ahr", f"{c_rate_peak_v:.3f} C" if np.isfinite(c_rate_peak_v) else "—")

    lv9, lv10, lv11 = st.columns(3, gap="small")
    with lv9:
        _metric_card("eff_c_rate_C = eff_c_rate<br>/ c_ref_ahr", f"{eff_c_rate_C_v:.3f} C" if np.isfinite(eff_c_rate_C_v) else "—")
    with lv10:
        _metric_card("dvdt_max_abs<br>(방전사이클 중 전압변화율의 최대값)", f"{dvdt_max_abs_v:.4f} V/s" if np.isfinite(dvdt_max_abs_v) else "—")
    with lv11:
        _metric_card("lli = slope(dC, dV)<br>/Voltage sensitivity to capacity loss (proxy)", f"{lli_v:.4f}" if np.isfinite(lli_v) else "—")

    st.markdown("#### Temperature")
    t1, t2, t3, t4 = st.columns(4, gap="small")
    with t1:
        _metric_card("실험조건 (실온/4도)", exp_cond_v)
    with t2:
        _metric_card("(Temperature_measured[0] at<br>discharge start, °C)", f"{temp_start_v:.2f} °C" if np.isfinite(temp_start_v) else "—")
    with t3:
        _metric_card("temperature_measured_max", f"{temp_max_v:.2f} °C" if np.isfinite(temp_max_v) else "—")
    with t4:
        _metric_card("temp_rise_cycle<br>(temperature_measured_max - Temperature_measured[0], °C)", f"{temp_rise_cycle_v:.2f} °C" if np.isfinite(temp_rise_cycle_v) else "—")

    t5, t6 = st.columns(2, gap="small")
    with t5:
        _metric_card("thermal_stresss<br>(exp(alpha * (Tmax - Tref)))", f"{thermal_stress_v:.3f}" if np.isfinite(thermal_stress_v) else "—")
    with t6:
        _metric_card("dTdt_max<br>(방전사이클 중 온도 상승률의 최대값)", f"{dTdt_max_v:.4f} °C/s" if np.isfinite(dTdt_max_v) else "—")

    st.markdown("#### Resistance")

    r1, r2, r3 = st.columns(3, gap="small")
    with r1:
        _metric_card("dcr(Re)", f"{dcr_v:.4f} Ω" if np.isfinite(dcr_v) else "—")
    with r2:
        _metric_card("dcr growth<br>(Internal resistance increase (log ratio vs early life))", f"{dcr_growth_v:.4f}" if np.isfinite(dcr_growth_v) else "—")
    with r3:
        _metric_card("lam = slope(dC, dRe)<br>/Resistance sensitivity to capacity loss (proxy)", f"{lam_v:.4f}" if np.isfinite(lam_v) else "—")


elif page == "Anomaly":
    st.markdown("## Anomaly")
    st.caption("Feature trajectories vs expected band (median ± IQR) and anomaly z-scores.")

    if df_feat_all is None or df_feat_all.empty:
        st.info("nasa_features_rul.csv 를 찾지 못했거나 비어 있습니다.")
    else:
        df_b = df_feat_all[df_feat_all["battery_id"] == str(selected_bid)].sort_values("cycle")
        if df_b.empty:
            st.info("선택한 배터리에 대한 cycle-feature row가 없습니다.")
        else:
            # Build bands
            bands = {}
            for feat in ["soh", "cap_pct", "dcr", "impedance_sum", "temperature_mean", "thermal_stress"]:
                if feat in df_feat_all.columns:
                    bands[feat] = _cycle_band_table(df_feat_all, feat)

            # Multi-row plot: (1) SoH/Cap% (2) DCR/Impedance (3) Temp + Thermal stress (right axis)
            figm = make_subplots(
                rows=3, cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                specs=[[{}], [{}], [{"secondary_y": True}]],
                row_heights=[0.34, 0.33, 0.33],
            )

            # Row 1: SoH + Cap%
            if "soh" in df_b.columns and "soh" in bands:
                band = bands["soh"]
                figm.add_trace(go.Scatter(x=band["cycle"], y=band["q1"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), row=1, col=1)
                figm.add_trace(go.Scatter(x=band["cycle"], y=band["q3"], mode="lines", line=dict(width=0), fill="tonexty", name="SoH expected band (IQR)"), row=1, col=1)
                figm.add_trace(go.Scatter(x=band["cycle"], y=band["median"], mode="lines", name="SoH expected (median)"), row=1, col=1)
                figm.add_trace(go.Scatter(x=df_b["cycle"], y=df_b["soh"], mode="lines", name="SoH (cell)"), row=1, col=1)

            if "cap_pct" in df_b.columns and "cap_pct" in bands:
                band = bands["cap_pct"]
                figm.add_trace(go.Scatter(x=band["cycle"], y=band["q1"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), row=1, col=1)
                figm.add_trace(go.Scatter(x=band["cycle"], y=band["q3"], mode="lines", line=dict(width=0), fill="tonexty", name="Cap% expected band (IQR)"), row=1, col=1)
                figm.add_trace(go.Scatter(x=band["cycle"], y=band["median"], mode="lines", name="Cap% expected (median)"), row=1, col=1)
                figm.add_trace(go.Scatter(x=df_b["cycle"], y=df_b["cap_pct"], mode="lines", name="Cap% (cell)"), row=1, col=1)

            # Row 2: DCR + Impedance
            if "dcr" in df_b.columns and "dcr" in bands:
                band = bands["dcr"]
                figm.add_trace(go.Scatter(x=band["cycle"], y=band["q1"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), row=2, col=1)
                figm.add_trace(go.Scatter(x=band["cycle"], y=band["q3"], mode="lines", line=dict(width=0), fill="tonexty", name="DCR expected band (IQR)"), row=2, col=1)
                figm.add_trace(go.Scatter(x=band["cycle"], y=band["median"], mode="lines", name="DCR expected (median)"), row=2, col=1)
                figm.add_trace(go.Scatter(x=df_b["cycle"], y=df_b["dcr"], mode="lines", name="DCR (cell)"), row=2, col=1)

            if "impedance_sum" in df_b.columns and "impedance_sum" in bands:
                band = bands["impedance_sum"]
                figm.add_trace(go.Scatter(x=band["cycle"], y=band["q1"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), row=2, col=1)
                figm.add_trace(go.Scatter(x=band["cycle"], y=band["q3"], mode="lines", line=dict(width=0), fill="tonexty", name="Impedance expected band (IQR)"), row=2, col=1)
                figm.add_trace(go.Scatter(x=band["cycle"], y=band["median"], mode="lines", name="Impedance expected (median)"), row=2, col=1)
                figm.add_trace(go.Scatter(x=df_b["cycle"], y=df_b["impedance_sum"], mode="lines", name="Impedance (cell)"), row=2, col=1)

            # Row 3: Temp + Thermal stress (right axis)
            if "temperature_mean" in df_b.columns and "temperature_mean" in bands:
                band = bands["temperature_mean"]
                figm.add_trace(go.Scatter(x=band["cycle"], y=band["q1"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), row=3, col=1, secondary_y=False)
                figm.add_trace(go.Scatter(x=band["cycle"], y=band["q3"], mode="lines", line=dict(width=0), fill="tonexty", name="Temp expected band (IQR)"), row=3, col=1, secondary_y=False)
                figm.add_trace(go.Scatter(x=band["cycle"], y=band["median"], mode="lines", name="Temp expected (median)"), row=3, col=1, secondary_y=False)
                figm.add_trace(go.Scatter(x=df_b["cycle"], y=df_b["temperature_mean"], mode="lines", name="Temp (cell)"), row=3, col=1, secondary_y=False)

            if "thermal_stress" in df_b.columns and "thermal_stress" in bands:
                band = bands["thermal_stress"]
                figm.add_trace(go.Scatter(x=band["cycle"], y=band["median"], mode="lines", name="Thermal stress expected (median)"), row=3, col=1, secondary_y=True)
                figm.add_trace(go.Scatter(x=df_b["cycle"], y=df_b["thermal_stress"], mode="lines", name="Thermal stress (cell)"), row=3, col=1, secondary_y=True)

            # Cursor vline
            for r in [1, 2, 3]:
                figm.add_vline(x=float(current_cycle), line_width=1, line_dash="dash", line_color="red", row=r, col=1)

            figm.update_layout(height=720, margin=dict(l=40, r=20, t=10, b=30))
            figm.update_xaxes(title_text="Cycle", row=3, col=1)
            figm.update_yaxes(title_text="SoH / Cap%", row=1, col=1)
            figm.update_yaxes(title_text="DCR / Impedance", row=2, col=1)
            figm.update_yaxes(title_text="Temp", row=3, col=1, secondary_y=False)
            figm.update_yaxes(title_text="Thermal stress", row=3, col=1, secondary_y=True)

            st.plotly_chart(figm, use_container_width=True)

            # --- Anomaly z-score plot (DCR and Cap%)
            st.markdown("### Anomaly z-score")
            z_df = pd.DataFrame({"cycle": df_b["cycle"].values})
            if "dcr" in df_b.columns and "dcr" in bands:
                z_df["z_dcr"] = _zscore_against_cycle_band(df_b, bands["dcr"], "dcr").values
            if "cap_pct" in df_b.columns and "cap_pct" in bands:
                z_df["z_cap_pct"] = _zscore_against_cycle_band(df_b, bands["cap_pct"], "cap_pct").values

            figz = go.Figure()
            for col in [c for c in z_df.columns if c.startswith("z_")]:
                figz.add_trace(go.Scatter(x=z_df["cycle"], y=z_df[col], mode="lines", name=col))
            figz.add_hline(y=2.0, line_dash="dash", line_color="gray")
            figz.add_hline(y=-2.0, line_dash="dash", line_color="gray")
            figz.add_vline(x=float(current_cycle), line_dash="dash", line_color="red")
            figz.update_layout(height=280, xaxis_title="Cycle", yaxis_title="z = (x - median)/IQR")
            st.plotly_chart(figz, use_container_width=True)

            # --- Evidence card (simple onset/severity)
            st.markdown("### Evidence cards")
            onset_dcr = -1
            sev_dcr = float("nan")
            if "z_dcr" in z_df.columns:
                onset_dcr = _first_onset_cycle(z_df["cycle"].to_numpy(), z_df["z_dcr"].to_numpy(), thr=2.0, consec=3)
                sev_dcr = float(np.nanmax(np.abs(z_df["z_dcr"].to_numpy()))) if z_df["z_dcr"].notna().any() else float("nan")

            onset_cap = -1
            sev_cap = float("nan")
            if "z_cap_pct" in z_df.columns:
                onset_cap = _first_onset_cycle(z_df["cycle"].to_numpy(), z_df["z_cap_pct"].to_numpy(), thr=2.0, consec=3)
                sev_cap = float(np.nanmax(np.abs(z_df["z_cap_pct"].to_numpy()))) if z_df["z_cap_pct"].notna().any() else float("nan")

            # choose main evidence feature
            main_feat = "z_dcr" if (not math.isnan(sev_dcr) and (math.isnan(sev_cap) or sev_dcr >= sev_cap)) else "z_cap_pct"
            main_onset = onset_dcr if main_feat == "z_dcr" else onset_cap
            main_sev = sev_dcr if main_feat == "z_dcr" else sev_cap

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Onset cycle (|z|>=2 for 3 cycles)", f"{main_onset}" if main_onset >= 0 else "N/A")
            with c2:
                st.metric("Severity (max |z|)", f"{main_sev:.2f}" if not math.isnan(main_sev) else "N/A")
            with c3:
                st.metric("Evidence feature", main_feat)

            # --- Report download (MD + simple PDF)
            st.markdown("### Report export")
            # markdown report
            report_md = f"""# Battery Monitoring Report\n\n- Battery: **{selected_bid}**\n- Current cycle: **{current_cycle}**\n- Pred RUL (cycles): **{current_pred_rul if not math.isnan(current_pred_rul) else 'N/A'}**\n- Risk: **{risk_lbl}**\n\n## Evidence\n- Onset cycle (main): **{main_onset if main_onset >= 0 else 'N/A'}**\n- Severity (max |z|): **{main_sev if not math.isnan(main_sev) else 'N/A'}**\n- Evidence feature: **{main_feat}**\n\n## Notes\n- z-score definition: z = (x - median(cycle)) / IQR(cycle)\n- Bands: median ± IQR (Q1–Q3)\n"""
            st.download_button(
                "Download report (MD)",
                data=report_md.encode("utf-8"),
                file_name=f"battery_report_{selected_bid}_cycle{current_cycle}.md",
                mime="text/markdown",
            )

            try:
                from reportlab.pdfgen import canvas as _pdf_canvas  # type: ignore
                from io import BytesIO
                buf = BytesIO()
                c = _pdf_canvas.Canvas(buf)
                y = 800
                c.setFont("Helvetica-Bold", 16)
                c.drawString(40, y, "Battery Monitoring Report")
                y -= 30
                c.setFont("Helvetica", 11)
                for line in [
                    f"Battery: {selected_bid}",
                    f"Current cycle: {current_cycle}",
                    f"Pred RUL (cycles): {current_pred_rul:.2f}" if not math.isnan(current_pred_rul) else "Pred RUL (cycles): N/A",
                    f"Risk: {risk_lbl}",
                    "",
                    "Evidence:",
                    f"  Onset cycle (main): {main_onset if main_onset >= 0 else 'N/A'}",
                    f"  Severity (max |z|): {main_sev:.2f}" if not math.isnan(main_sev) else "  Severity (max |z|): N/A",
                    f"  Evidence feature: {main_feat}",
                    "",
                    "z = (x - median(cycle)) / IQR(cycle), band = Q1–Q3",
                ]:
                    c.drawString(40, y, line)
                    y -= 16
                    if y < 60:
                        c.showPage()
                        y = 800
                        c.setFont("Helvetica", 11)
                c.showPage()
                c.save()
                pdf_bytes = buf.getvalue()
                st.download_button(
                    "Download report (PDF)",
                    data=pdf_bytes,
                    file_name=f"battery_report_{selected_bid}_cycle{current_cycle}.pdf",
                    mime="application/pdf",
                )
            except Exception:
                st.caption("PDF export unavailable (reportlab not installed).")

# -----------------------------
# COMPARE / FLEET TAB
# -----------------------------
elif page == "Compare / Fleet":
    st.markdown("## Compare / Fleet view")
    st.caption("Rank batteries by anomaly z-scores at the current cycle (z(DCR), z(Cap%)).")

    pick = st.multiselect(
        "Select up to 4 batteries",
        options=all_bids,
        default=[b for b in ["B0018", "B0042", "B0043", "B0033"] if b in all_bids][:4],
        max_selections=4,
    )

    if df_feat_all is None or df_feat_all.empty:
        st.info("No cycle-feature table available.")
    elif not pick:
        st.info("Select at least one battery.")
    else:
        # Compute per-cycle band for the cursor cycle only
        cur_cyc = float(current_cycle)
        df_cur = df_feat_all[df_feat_all["cycle"] == cur_cyc]
        rows = []
        # precompute median/iqr at this cycle
        def _med_iqr(df, col):
            if col not in df.columns or df.empty:
                return (float("nan"), float("nan"))
            med = float(df[col].median())
            q1 = float(df[col].quantile(0.25))
            q3 = float(df[col].quantile(0.75))
            iqr = q3 - q1
            if iqr == 0:
                iqr = float("nan")
            return (med, iqr)

        dcr_med, dcr_iqr = _med_iqr(df_cur, "dcr")
        cap_med, cap_iqr = _med_iqr(df_cur, "cap_pct")

        for bid in pick:
            sub = df_feat_all[(df_feat_all["battery_id"] == str(bid)) & (df_feat_all["cycle"] == cur_cyc)]
            if sub.empty:
                rows.append({"battery_id": bid, "z_dcr": np.nan, "z_cap_pct": np.nan, "risk": np.nan})
                continue
            dcr = _safe_float(sub.iloc[-1]["dcr"]) if "dcr" in sub.columns else float("nan")
            cap = _safe_float(sub.iloc[-1]["cap_pct"]) if "cap_pct" in sub.columns else float("nan")
            z_dcr = (dcr - dcr_med) / dcr_iqr if (not math.isnan(dcr) and not math.isnan(dcr_med) and not math.isnan(dcr_iqr)) else float("nan")
            z_cap = (cap - cap_med) / cap_iqr if (not math.isnan(cap) and not math.isnan(cap_med) and not math.isnan(cap_iqr)) else float("nan")
            # risk as max |z|
            risk = float(np.nanmax(np.abs([z_dcr, z_cap])))
            rows.append({"battery_id": bid, "z_dcr": z_dcr, "z_cap_pct": z_cap, "risk": risk})

        df_rank = pd.DataFrame(rows).sort_values("risk", ascending=False)
        st.dataframe(df_rank, use_container_width=True, height=240)

        st.caption("Drill-down: pick Battery ID in the left sidebar to inspect it in Overview/Monitoring.")

# -----------------------------
# WHAT-IF TAB (Scenario builder)
# -----------------------------

else:  # What-if
    st.markdown("## What-if")
    st.caption("현재 버전에서는 Overview/Anomaly 안정화에 집중하기 위해 Scenario builder를 잠시 비활성화했습니다.")
