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

import requests
import smtplib
from email.message import EmailMessage

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

BMAML_DIR = FILE_DIR / "dashboard_export_v2" / "bmaml_v2"
CKPT_DEFAULT = FILE_DIR / "core_checkpoints" / "nasa_bmaml_best_re.pt"
SHAP_JSON = FILE_DIR / "shap_outputs" / "bmaml_shap_seq_feature_importance.json"

def resolve_data_path(filename: str) -> Path:
    """Resolve data file path across common project layouts (analysis/, repo root, CWD)."""
    candidates = [
        FILE_DIR / "analysis" / filename,
        FILE_DIR / filename,
        Path.cwd() / filename,
        Path.cwd() / "analysis" / filename,
    ]
    # sandbox fallback
    md = Path("/mnt/data") / filename
    candidates.append(md)
    for p in candidates:
        if p.exists():
            return p
    # default to analysis/ for stable relative outputs
    return candidates[0]

FEATURE_STATS_JSON_PATH = resolve_data_path("feature_rul_stats.json")
FEATURE_STATS_CSV_PATH = resolve_data_path("feature_rul_stats.csv")

# Prefer CSV if available (easier to inspect/version)
FEATURE_STATS_PATH = FEATURE_STATS_CSV_PATH if FEATURE_STATS_CSV_PATH.exists() else FEATURE_STATS_JSON_PATH
NASA_FEATURES_PATH = resolve_data_path("nasa_features_rul.csv")

EXPORT_TEST_BATTERIES = {"B0018", "B0033", "B0043", "B0055"}  # 첫 화면용

# (legacy) preferred list was hard-coded; keep initial demo batteries only
preferred = sorted(list(EXPORT_TEST_BATTERIES))

DEFAULT_R_RATIO = 0.25
DEFAULT_CYCLES_PER_DAY = 0.5  # default average cycles/day

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
AUTO_PLAY_STEP = 1          # 한 번에 몇 Cycle씩 움직일지

# -------------------------------------------------
# 시나리오 빌더에 사용할 feature 정의
# (feature_rul_stats.json + causal 피처 기준)
# -------------------------------------------------
SCENARIO_FEATURES = {
    # Core degradation state
    "soh": {"label": "SoH (state of health, 0–1)", "fallback_min": 0.6, "fallback_max": 1.0, "step": 0.01},
    "capacity_mean": {"label": "Mean discharge capacity (Ah)", "fallback_min": 1.0, "fallback_max": 3.5, "step": 0.05},

    # Electrical features
    "dcr": {"label": "DCR proxy: Re (electrolyte resistance, Ω) (stored as dcr)", "fallback_min": 0.0, "fallback_max": 1.0, "step": 0.001},
    "impedance_sum": {"label": "Impedance sum (Re + Rct, Ω)", "fallback_min": 0.0, "fallback_max": 2.0, "step": 0.001},

    # Mechanism proxies
    "lli": {"label": "LLI proxy (loss of lithium inventory, a.u.)", "fallback_min": -50.0, "fallback_max": 50.0, "step": 1.0},
    "lam": {"label": "LAM proxy (loss of active material, a.u.)", "fallback_min": -50.0, "fallback_max": 50.0, "step": 1.0},

    # Operating / conditions
    "ambient_temp_c": {"label": "Discharge Temperature_measured start (cell temp start, °C)", "fallback_min": 10.0, "fallback_max": 45.0, "step": 1.0},
    "temperature_mean": {"label": "Mean cell temperature during discharge (°C)", "fallback_min": 10.0, "fallback_max": 60.0, "step": 1.0},
    "temp_rise_cycle": {"label": "Temp rise within discharge (Tmax − Tstart, °C)", "fallback_min": 0.0, "fallback_max": 20.0, "step": 0.5},
    "thermal_stress": {"label": "Thermal stress index (a.u.)", "fallback_min": 0.0, "fallback_max": 3.0, "step": 0.05},

    # Voltage/current summaries (optional knobs)
    "voltage_min": {"label": "Minimum voltage during discharge (V)", "fallback_min": 2.0, "fallback_max": 3.5, "step": 0.02},
    "voltage_measured_mean": {"label": "Mean terminal voltage during discharge (V)", "fallback_min": 2.5, "fallback_max": 4.2, "step": 0.02},
    "current_mean": {"label": "Mean current during discharge (A, negative = discharge)", "fallback_min": -6.0, "fallback_max": 0.0, "step": 0.1},
    "current_min": {"label": "Peak discharge current (A, most negative)", "fallback_min": -10.0, "fallback_max": 0.0, "step": 0.2},

    # Derived capacity-loss style knob
    "regen_strength": {"label": "Capacity drop vs best-so-far (Ah) (regen_strength)", "fallback_min": 0.0, "fallback_max": 1.5, "step": 0.05},
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
            # 새 배터리/Cycle이면 슬라이더도 Baseline으로 강제 이동
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
        Reset scenario 버튼에서 사용:
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
    - 배터리/Cycle 컨텍스트(bid, cycle)가 바뀌면
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
        현재 배터리/Cycle 컨텍스트에서
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
        "Cycle 동안 온도 상승이 작을수록 유리합니다. 냉각을 잘 해서 온도 차이를 줄이는 방향으로 가정하세요."
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
    """nasa_features_rul.csv: per-battery, per-cycle feature table."""
    if not NASA_FEATURES_PATH.exists():
        return None
    df = pd.read_csv(NASA_FEATURES_PATH)

    # Column normalization (avoid naming drift across exports)
    col_aliases = {
    # id / index
    "battery": "battery",
    "battery_id": "battery_id",
    "cycle_num": "cycle",
    "cycle_index": "cycle",
    # temps
    "ambient_temperature": "ambient_temp_c",
    "ambient_temp": "ambient_temp_c",
    "ambient_temp_C": "ambient_temp_c",
    "Temperature_measured_mean": "temperature_mean",
    "temp_mean": "temperature_mean",
    "Temp_mean": "temperature_mean",
    # impedance / resistance
    "dcr_ohm": "dcr",
    "dc_resistance": "dcr",
    "impedance_re_rct": "impedance_sum",
    "impedance_sum_ohm": "impedance_sum",
    # capacity / SoH
    "soh_pct": "soh",
    "cap_retention": "cap_pct",
    "capacity_pct": "cap_pct",
    "thermal_stress_idx": "thermal_stress",
    }
    df.rename(columns={k: v for k, v in col_aliases.items() if k in df.columns}, inplace=True)
    # Column normalization for identifiers
    if "battery_id" not in df.columns:
        if "battery" in df.columns:
            df["battery_id"] = df["battery"].astype(str)
        else:
            # fallback: try common id columns
            for c in ["bid", "batteryID", "cell_id"]:
                if c in df.columns:
                    df["battery_id"] = df[c].astype(str)
                    break
    else:
        df["battery_id"] = df["battery_id"].astype(str)

    if "cycle" not in df.columns:
        for c in ["cycle_num", "cycle_index", "cycleId"]:
            if c in df.columns:
                df.rename(columns={c: "cycle"}, inplace=True)
                break

    return df


class CycleFeatureContext:
    """현재 배터리/Cycle 메타 피처 + 파이프라인 SoH 래핑."""

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
        with st.spinner("Loading meta-learner & battery tasks..."):
            meta_state = load_meta_state(str(CKPT_DEFAULT), eval_dataset="from_ckpt")
            records_rt = build_runtime_records(meta_state=meta_state, r_ratio=r_ratio)

        st.session_state["records"] = records_rt
        st.session_state["r_ratio"] = float(r_ratio)
        st.session_state["records_source"] = f"runtime (r_ratio={r_ratio:.2f})"
        loading_placeholder.empty()

    except Exception as e:
        loading_placeholder.empty()
        st.error(
            "Failed to create runtime records.\n"
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
st.title("👁️🔋 Real-time Li-ion Battery RUL Prediction Dashboard")
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
            f"Could not find precomputed JSON for DEFAULT_R_RATIO={DEFAULT_R_RATIO:.2f}.\n"
            "export_rul_dashboard_data_meta_fixed.py 를 이 r_ratio로 먼저 돌려 주세요."
        )
        st.stop()

st.sidebar.header("1) Model settings")

# 체크포인트 없어도 precomputed는 쓸 수 있게
ckpt_exists = CKPT_DEFAULT.exists()

if not RUNTIME_AVAILABLE:
    st.sidebar.info(
        "Runtime dependencies (e.g., torch) are missing, so only **Fast load (precomputed)** is available.\n"
        f"(import error: {_RUNTIME_IMPORT_ERROR})"
    )
if not ckpt_exists:
    st.sidebar.warning(
        f"checkpoint not found:\n{CKPT_DEFAULT}\n"
        "→ Live re-adaptation is unavailable; use **Fast load (precomputed)** below."
    )

# 🔽 슬라이더 기본값을 0.15~0.40 사이로 클램프
slider_default = float(
    np.clip(st.session_state.get("r_ratio", DEFAULT_R_RATIO), 0.0, 0.35)
)

r_ratio = st.sidebar.slider(
    "Initial adaptation ratio (r_ratio)",
    min_value=0.0,
    max_value=0.35,
    value=slider_default,
    step=0.05,
    help="Controls what fraction of early cycles is used as the initial adaptation window.",
)

# ❗ 이 문구 유지
st.sidebar.caption(
    "Tip: to predict with less initial data, lower the ratio a bit (e.g., 0.25 → 0.20)."
)

# 1) 재추론 버튼
init_clicked = st.sidebar.button(
    "Initialize model & run inference",
    disabled=(not ckpt_exists) or (not RUNTIME_AVAILABLE),
)

# 재추론 안내 문구
st.sidebar.caption(
    "Inference may take ~1 minute. For a quick start, use Fast load below."
)

# 2) 빠른 로드 버튼 (precomputed)
fast_load_clicked = st.sidebar.button("Fast load (precomputed)")

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
            f"Loaded precomputed RUL results for r_ratio={target_r:.2f}.\n"
            f"Applied batteries: {loaded_bids}"
        )
    else:
        st.sidebar.error(
            f"Could not find precomputed JSON for r_ratio={target_r:.2f}.\n"
            "Check the export path / filenames."
        )
else:
    st.session_state["force_recompute"] = False


# -----------------------------
# -----------------------------
# Notification config is shown in Compare / Fleet (kept out of the sidebar).
# -----------------------------
st.session_state.setdefault('enable_notif', False)
st.session_state.setdefault('slack_webhook_url', '')
st.session_state.setdefault('email_enabled', False)
st.session_state.setdefault('smtp_host', '')
st.session_state.setdefault('smtp_port', 587)
st.session_state.setdefault('smtp_user', '')
st.session_state.setdefault('smtp_pass', '')
st.session_state.setdefault('from_email', '')
st.session_state.setdefault('to_email', '')

# =================================================
# Main content (tabs)
# =================================================
tab_overview, tab_monitoring, tab_compare, tab_whatif = st.tabs(
    ["Overview", "Monitoring", "Compare / Fleet", "What-if"]
)

# -----------------------------
# Helpers for Monitoring / Compare
# -----------------------------

# -----------------------------
# Alerts (Slack / Email)
# -----------------------------
def _send_slack(webhook_url: str, message: str) -> Tuple[bool, str]:
    try:
        if not webhook_url:
            return False, "Slack webhook URL is empty."
        resp = requests.post(webhook_url, json={"text": message}, timeout=8)
        if 200 <= resp.status_code < 300:
            return True, "OK"
        return False, f"Slack HTTP {resp.status_code}: {resp.text[:200]}"
    except Exception as e:
        return False, f"Slack error: {e}"

def _send_email_smtp(
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    to_email: str,
    subject: str,
    body: str,
    use_tls: bool = True,
) -> Tuple[bool, str]:
    try:
        if not (smtp_host and smtp_port and smtp_user and smtp_password and to_email):
            return False, "Email settings are incomplete."
        msg = EmailMessage()
        msg["From"] = smtp_user
        msg["To"] = to_email
        msg["Subject"] = subject
        msg.set_content(body)

        if use_tls:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=12) as s:
                s.starttls()
                s.login(smtp_user, smtp_password)
                s.send_message(msg)
        else:
            with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=12) as s:
                s.login(smtp_user, smtp_password)
                s.send_message(msg)
        return True, "OK"
    except Exception as e:
        return False, f"Email error: {e}"

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
with tab_overview:
    left, right = st.columns([1.25, 1.75], gap="large")

    # --- Left: key metrics + quick degradation snapshot
    with left:
        st.markdown(f"## Battery {selected_bid}")

        if str(selected_bid) == "B0043":
            st.caption("After EOL / severe degradation, feature distribution shifts (e.g., temperature/self-heating) can push the model out of its training manifold, causing non-physical RUL rebounds. Treat predictions in this region as unreliable.")

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

        
    with right:
        st.markdown("## RUL trajectory")
        fig = go.Figure()

        # 1) True RUL: 전체 구간 (offline reference)
        fig.add_trace(
            go.Scatter(
                x=cycles,
                y=rul_true,
                mode="lines",
                name="RUL (true)",
                line=dict(color="rgba(0,0,0,0.5)", dash="dash"),
            )
        )

        # 2) Prediction + uncertainty from current cycle onward
        if in_future_region and fut_cycles.size > 0:
            mask_future = fut_cycles >= float(current_cycle)
            fut_x = fut_cycles[mask_future]
            fut_pred_y = fut_rul_pred[mask_future]
            fut_std_y = fut_rul_std[mask_future]

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
        if cycles.size > 0 and rul_true.size == cycles.size:
            eol_idxs = np.where(rul_true <= 0)[0]
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
        st.plotly_chart(fig, use_container_width=True)
# -----------------------------
# MONITORING TAB
# -----------------------------
with tab_monitoring:
    st.markdown(f"## Monitoring — {selected_bid}")
    st.caption("Read anomalies fast. Each plot shows the selected cell vs fleet expected band (Q1–Q3) per cycle.")

    if df_feat_all is None or df_feat_all.empty:
        st.info("Could not find nasa_features_rul.csv (or it is empty).")
    else:
        df_b = df_feat_all[df_feat_all["battery_id"] == str(selected_bid)].sort_values("cycle")
        if df_b.empty:
            st.info("No cycle-feature rows found for the selected battery.")
        else:
            with st.expander("How is the expected band computed?", expanded=False):
                st.markdown(
                    "- For each **cycle**, aggregate values across **all batteries**.\n"
                    "- **Expected median** = fleet median at that cycle.\n"
                    "- **Expected band** = [Q1, Q3] at that cycle (IQR band).\n"
                    "- **Out-of-band** = cell value < Q1 or > Q3 at that cycle."
                )

            core_feats = [
                ("soh", "SoH (0–1)"),
                ("cap_pct", "Capacity retention (%)"),
                ("dcr", "DCR (Ω)"),
                ("impedance_sum", "Impedance (Re+Rct, Ω)"),
                ("temperature_mean", "Mean cell temp (°C)"),
                ("thermal_stress", "Thermal stress (a.u.)"),
                ("lli", "LLI proxy (a.u.)"),
                ("lam", "LAM proxy (a.u.)"),
            ]

            # Build cycle bands
            bands = {}
            for feat, _ in core_feats:
                if feat in df_feat_all.columns:
                    bands[feat] = _cycle_band_table(df_feat_all, feat)

            # z-scores for all available features
            z_df = df_b[["cycle"]].copy()
            for feat, _ in core_feats:
                if feat in df_b.columns and feat in bands:
                    z_df[f"z_{feat}"] = _zscore_against_cycle_band(df_b, bands[feat], feat).values

            plot_col, info_col = st.columns([3.2, 1.25], gap="large")

            with info_col:
                st.markdown("### Legend")
                st.markdown(
                    "- **Colored solid**: selected cell\n"
                    "- **Gray dashed**: fleet median\n"
                    "- **Gray band**: fleet Q1–Q3\n"
                    "- **Red dots**: out-of-band points"
                )
                st.markdown("### APM thresholds")
                st.markdown(
                    "- **WARN**: |z| ≥ 2 (sustained)\n"
                    "- **ALERT**: |z| ≥ 3 (sustained)\n"
                    "- Severity = max(|z|) at the current cycle"
                )

            def _add_band_and_cell(fig, feat: str, label: str, cell_color: str):
                if feat not in df_b.columns or feat not in bands:
                    return
                b = bands[feat]
                merged = pd.merge(
                    df_b[["cycle", feat]].rename(columns={feat: "x"}),
                    b[["cycle", "q1", "q3", "median"]],
                    on="cycle",
                    how="left",
                )
                # band (IQR)
                fig.add_trace(go.Scatter(x=merged["cycle"], y=merged["q1"], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False))
                fig.add_trace(go.Scatter(x=merged["cycle"], y=merged["q3"], mode="lines", line=dict(width=0), fill="tonexty",
                                         fillcolor="rgba(0,0,0,0.10)", hoverinfo="skip", showlegend=False))
                # median
                fig.add_trace(go.Scatter(x=merged["cycle"], y=merged["median"], mode="lines",
                                         line=dict(color="rgba(0,0,0,0.45)", dash="dash"), showlegend=False))
                # cell
                fig.add_trace(go.Scatter(x=merged["cycle"], y=merged["x"], mode="lines",
                                         line=dict(color=cell_color, width=2), name=label))
                # out-of-band
                oob = (merged["x"] < merged["q1"]) | (merged["x"] > merged["q3"])
                if oob.any():
                    fig.add_trace(go.Scatter(x=merged.loc[oob, "cycle"], y=merged.loc[oob, "x"], mode="markers",
                                             marker=dict(size=6, color="rgba(220,0,0,0.75)"), showlegend=False))

            with plot_col:
                st.markdown("### SoH & Capacity")
                fig1 = go.Figure()
                _add_band_and_cell(fig1, "soh", "SoH", "#2ca02c")
                _add_band_and_cell(fig1, "cap_pct", "Cap%", "#ff7f0e")
                fig1.add_vline(x=float(current_cycle), line_dash="dash", line_color="red")
                fig1.update_layout(height=320, margin=dict(l=30, r=20, t=10, b=35), xaxis_title="Cycle", yaxis_title="",
                                   legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
                st.plotly_chart(fig1, use_container_width=True)

                st.markdown("### Resistance (Re) & Impedance")
                fig2 = go.Figure()
                _add_band_and_cell(fig2, "dcr", "Re (dcr)", "#1f77b4")
                _add_band_and_cell(fig2, "impedance_sum", "Impedance", "#9467bd")
                fig2.add_vline(x=float(current_cycle), line_dash="dash", line_color="red")
                fig2.update_layout(height=320, margin=dict(l=30, r=20, t=10, b=35), xaxis_title="Cycle", yaxis_title="",
                                   legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
                st.plotly_chart(fig2, use_container_width=True)

                st.markdown("### Temperature & Thermal stress")
                fig3 = make_subplots(specs=[[{"secondary_y": True}]])
                if "temperature_mean" in df_b.columns and "temperature_mean" in bands:
                    tmp = go.Figure()
                    _add_band_and_cell(tmp, "temperature_mean", "Temp (mean)", "#d62728")
                    for tr in tmp.data:
                        fig3.add_trace(tr, secondary_y=False)
                if "thermal_stress" in df_b.columns and "thermal_stress" in bands:
                    b = bands["thermal_stress"]
                    merged = pd.merge(
                        df_b[["cycle", "thermal_stress"]].rename(columns={"thermal_stress": "x"}),
                        b[["cycle", "median"]],
                        on="cycle",
                        how="left",
                    )
                    fig3.add_trace(go.Scatter(x=merged["cycle"], y=merged["median"], mode="lines",
                                              line=dict(color="rgba(0,0,0,0.35)", dash="dash"), showlegend=False),
                                   secondary_y=True)
                    fig3.add_trace(go.Scatter(x=merged["cycle"], y=merged["x"], mode="lines",
                                              line=dict(color="#8c564b", width=2), name="Thermal stress"),
                                   secondary_y=True)
                fig3.add_vline(x=float(current_cycle), line_dash="dash", line_color="red")
                fig3.update_layout(height=320, margin=dict(l=30, r=40, t=10, b=35), xaxis_title="Cycle", yaxis_title="",
                                   legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
                fig3.update_yaxes(title_text="", secondary_y=False)
                fig3.update_yaxes(title_text="", secondary_y=True)
                st.plotly_chart(fig3, use_container_width=True)

                st.markdown("## Anomaly z-score")
                z_cols = [c for c in z_df.columns if c.startswith("z_")]
                default_cols = [c for c in ["z_dcr", "z_cap_pct"] if c in z_cols]
                pick_z = st.multiselect("Signals", options=z_cols, default=default_cols if default_cols else z_cols[:2])

                figz = go.Figure()
                red_palette = ["#8b0000", "#b22222", "#dc143c", "#ff4500", "#ff6347", "#ff7f7f"]
                for k, col in enumerate(pick_z):
                    figz.add_trace(go.Scatter(x=z_df["cycle"], y=z_df[col], mode="lines", name=col,
                                              line=dict(color=red_palette[k % len(red_palette)], width=2)))
                for thr, dash in [(2.0, "dash"), (-2.0, "dash"), (3.0, "dot"), (-3.0, "dot")]:
                    figz.add_hline(y=thr, line_dash=dash, line_color="rgba(0,0,0,0.45)")
                figz.add_vline(x=float(current_cycle), line_dash="dash", line_color="red")
                figz.update_layout(height=300, xaxis_title="Cycle", yaxis_title="", margin=dict(l=30, r=20, t=10, b=35))
                st.plotly_chart(figz, use_container_width=True)

                st.markdown("### Evidence cards")
                evid = []
                for col in pick_z:
                    arr = z_df[col].to_numpy(dtype=float)
                    cyc = z_df["cycle"].to_numpy(dtype=float)
                    onset = _first_onset_cycle(cyc, arr, thr=2.0, consec=3)
                    sev = float(np.nanmax(np.abs(arr))) if np.isfinite(arr).any() else float("nan")
                    evid.append({"signal": col, "onset_cycle(|z|≥2 for 3x)": onset if onset >= 0 else None, "peak_|z|": sev})
                if pick_z:
                    df_now = z_df[z_df["cycle"] == float(current_cycle)]
                    if not df_now.empty:
                        vals = [abs(_safe_float(df_now.iloc[-1][c])) for c in pick_z]
                        severity_now = float(np.nanmax(vals)) if vals else float("nan")
                        st.metric("Current severity (max |z|)", f"{severity_now:.2f}" if not math.isnan(severity_now) else "N/A")
                st.dataframe(pd.DataFrame(evid), use_container_width=True, hide_index=True)
# COMPARE / FLEET TAB
# -----------------------------
with tab_compare:
    st.markdown("## Compare / Fleet view")
    st.caption("Rank batteries by anomaly z-scores at the current cycle (z(Re=dcr), z(Cap%)).")

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
        st.markdown("### Feature-wise anomaly compare")
        # Compare per-feature z-score/out-of-band for the selected batteries at the cursor cycle
        cand_feats = [f for f in ["dcr","cap_pct","impedance_sum","temperature_mean","soh","lli","lam","thermal_stress"] if f in df_feat_all.columns]
        if cand_feats:
            feat_pick = st.selectbox("Feature", options=cand_feats, index=0)
            med = float(df_cur[feat_pick].median()) if not df_cur.empty else float("nan")
            q1 = float(df_cur[feat_pick].quantile(0.25)) if not df_cur.empty else float("nan")
            q3 = float(df_cur[feat_pick].quantile(0.75)) if not df_cur.empty else float("nan")
            iqr = (q3 - q1)
            if iqr == 0:
                iqr = float("nan")

            rows2 = []
            for bid in pick:
                sub = df_feat_all[(df_feat_all["battery_id"] == str(bid)) & (df_feat_all["cycle"] == cur_cyc)]
                x = _safe_float(sub.iloc[-1][feat_pick]) if (not sub.empty and feat_pick in sub.columns) else float("nan")
                z = (x - med) / iqr if (not math.isnan(x) and not math.isnan(med) and not math.isnan(iqr)) else float("nan")
                oob = (not math.isnan(x) and not math.isnan(q1) and not math.isnan(q3) and (x < q1 or x > q3))
                rows2.append({"battery_id": bid, "value": x, "z": z, "out_of_band(Q1–Q3)": oob})
            df2 = pd.DataFrame(rows2).sort_values("z", key=lambda s: s.abs(), ascending=False)
            st.dataframe(df2, use_container_width=True, hide_index=True)

        st.markdown("### Notifications (Slack / Email)")
        with st.expander("Configure notifications", expanded=False):
            st.caption("Triggers when the Overview risk badge becomes WARN/ALERT (dedup by risk and cycle).")
            enable_notif = st.checkbox("Enable notifications", value=st.session_state.get("enable_notif", False), key="enable_notif")

            st.markdown("**Slack**")
            slack_webhook_url = st.text_input("Slack Incoming Webhook URL", value=st.session_state.get("slack_webhook_url", ""), key="slack_webhook_url")

            st.markdown("**Email (SMTP)**")
            email_enabled = st.checkbox("Enable email", value=st.session_state.get("email_enabled", False), key="email_enabled")
            c1, c2 = st.columns(2)
            with c1:
                smtp_host = st.text_input("SMTP host", value=st.session_state.get("smtp_host", ""), key="smtp_host", disabled=not email_enabled)
                smtp_user = st.text_input("SMTP user", value=st.session_state.get("smtp_user", ""), key="smtp_user", disabled=not email_enabled)
                from_email = st.text_input("From", value=st.session_state.get("from_email", ""), key="from_email", disabled=not email_enabled)
            with c2:
                smtp_port = st.number_input("SMTP port", min_value=1, max_value=65535, value=int(st.session_state.get("smtp_port", 587)), step=1, key="smtp_port", disabled=not email_enabled)
                smtp_pass = st.text_input("SMTP password", value=st.session_state.get("smtp_pass", ""), key="smtp_pass", type="password", disabled=not email_enabled)
                to_email = st.text_input("To", value=st.session_state.get("to_email", ""), key="to_email", disabled=not email_enabled)

            if st.button("Send test notification", disabled=not enable_notif):
                test_msg = "Test notification from the RUL dashboard."
                ok_s, msg_s = (False, "Slack disabled")
                ok_e, msg_e = (False, "Email disabled")
                if slack_webhook_url:
                    ok_s, msg_s = send_slack_webhook(slack_webhook_url, test_msg)
                if email_enabled:
                    ok_e, msg_e = send_email_smtp(
                        host=smtp_host, port=int(smtp_port), user=smtp_user, password=smtp_pass,
                        from_email=from_email, to_email=to_email,
                        subject="[RUL Dashboard] Test", body=test_msg
                    )
                st.success(f"Slack: {ok_s} ({msg_s}) | Email: {ok_e} ({msg_e})")

        st.caption("Drill-down: pick Battery ID in the left sidebar to inspect it in Overview/Monitoring.")

# -----------------------------
# WHAT-IF TAB (Scenario builder)
# -----------------------------
with tab_whatif:
    st.markdown("## What-if")
    st.caption("To run policy-knob what-if analysis, enable the Scenario Builder here and adjust the knobs.")

    with st.expander("Scenario builder (fold/unfold)", expanded=True):
        sb_left, sb_right = st.columns([2.0, 2.0], gap="large")

# BOTTOM-LEFT: Usage & Scenario input
# =================================================
with sb_left:
    st.markdown("### Scenario builder")
    st.caption("Note: sliders apply what-if adjustments only at the currently selected cycle.")

    cycles_per_day_sb = st.number_input(
        "Average cycles driven per day",
        min_value=0.1,
        max_value=5.0,
        value=st.session_state.get("cycles_per_day", DEFAULT_CYCLES_PER_DAY),
        step=0.1,
        key="cycles_per_day",
    )

    if FEATURE_STATS is None:
        st.warning("feature_rul_stats.json not found. Run the export script first.")
    else:
        # 🔒 자동 재생 중에는 시나리오 슬라이더/계산을 건드리지 않음
        if st.session_state.get("auto_play", False):
            st.info(
                "Scenario Builder is paused while auto-play is running.\n\n"
            )
        elif math.isnan(current_pred_rul):
            st.info("This cycle is outside the prediction region (no prediction).\n\nMove the cycle cursor past the r_ratio boundary to enable Scenario calculation.")
        else:
            st.markdown("#### Baseline vs Scenario (degradation conditions)")
            st.caption(
                "At the current battery/cycle, Baseline (A) is the reference, and you can change Scenario (B) to see how RUL shifts. "
                ""
            )

            # Allow what-if on environmental conditions (e.g., ambient_temp_c=43°C)
            lock_env_features = st.checkbox(
                "Lock environmental features (ambient_temp_c, temp_rise_cycle)",
                value=True,
                help="Uncheck to edit ambient_temp_c / temp_rise_cycle in the sliders.",
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

            # 🔹 현재 배터리/Cycle 기준 Scenario 컨텍스트 생성
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

                    # 👉 현재 배터리/Cycle 피처값을 Baseline 기본값으로 사용
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
                        is_locked = lock_env_features and feat_key in {"ambient_temp_c", "temp_rise_cycle"}

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
with sb_right:
    st.markdown("### Scenario results")

    scenario_inputs = st.session_state.get("scenario_inputs", None)

    # 🔁 Reset scenario 버튼
    reset_clicked = st.button(
        "Reset scenario",
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
            "Scenario results are not updated during auto-play. "
            "자동 재생을 끄고 시나리오를 조정해 주세요."
        )
    elif scenario_inputs is None or FEATURE_STATS is None:
        st.info("Set baseline / scenario conditions on the left first.")
    else:
        baseline = scenario_inputs["baseline"]          # dict: feat_key -> baseline 값
        scenario = scenario_inputs["scenario"]          # dict: feat_key -> scenario 값
        cycles_per_day_sb = scenario_inputs["cycles_per_day"]

        current_pred_rul_s = float(scenario_inputs["current_pred_rul"])
        current_true_rul_s = float(scenario_inputs["current_true_rul"])

        if math.isnan(current_pred_rul_s):
            st.info(
                """현재 Cycle은 예측 구간이 아니라 Scenario 결과를 계산할 수 없습니다. (예측 없음)

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
        with st.expander("What changed & how it affects RUL (quick summary)", expanded=True):
            if not any_changed:
                st.caption("No features were changed in this scenario.")
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
        add_clicked = st.button("Add current settings to the scenario table")

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
                f"Scenario added: {current_pred_rul_s:.1f} → {scen_rul:.1f} cycles "
                f"({diff_cycles:+.1f} cycles)"
            )

        # 5) 누적 시나리오 테이블
        if st.session_state["scenarios"]:
            st.markdown("#### Cumulative scenario table")

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

st.markdown(
"""
<div style="font-size:13px; line-height:1.6; color:#222; max-width:860px; margin:0 auto;">
<b>Overview</b>: RUL trajectory(±2σ) + current cycle cursor + quick anomaly z-scores.<br/>
<b>Monitoring</b>: Feature trajectory vs expected band + z-score + evidence cards + report export.<br/>
<b>Compare / Fleet</b>: Up to 4 batteries ranked by z(DCR), z(Cap%) at the current cycle.<br/>
<b>What-if</b>: Scenario builder (fold/unfold) for slider-based what-if.<br/><br/>
</div>
""",
unsafe_allow_html=True,
)