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
import streamlit as st
import glob
from pathlib import Path
from deep_learning.core.rul_precomputed_loader_restored import PrecomputedRULLoader


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

FEATURE_STATS_PATH = FILE_DIR / "analysis" / "feature_rul_stats.json"
NASA_FEATURES_PATH = FILE_DIR / "analysis" / "nasa_features_rul.csv"

EXPORT_TEST_BATTERIES = {"B0018", "B0033", "B0043", "B0055"}  # 첫 화면용

preferred = ["B0018", "B0033", "B0043", "B0055"]

DEFAULT_R_RATIO = 0.25
DEFAULT_CYCLES_PER_DAY = 0.5  # 좀 더 현실적인 기본값

from deep_learning.core.prefix_inference_viz_meta_restored import (
    build_model_and_grouped,
    make_task_prefix,
    run_adapt_and_predict,
)
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
    """export_nasa_feature_rul_stats.py가 만든 feature_rul_stats.json 로드."""
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


FEATURE_STATS = load_feature_stats(FEATURE_STATS_PATH)
shap_names, shap_vals = load_shap_importance(SHAP_JSON)

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

    # 두 번째 추론에서는 33번 배터리는 제외 (eval 셋 기준)
    records.pop("B0033", None)
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
            st.image(str(LOADING_GIF), use_container_width=True)

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

MODEL_TAG = "Few-shot physics-informed meta learninng RUL, transformer backbone)"
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
    disabled=not ckpt_exists,
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
preferred = ["B0018", "B0033", "B0050", "B0042"]
all_bids = [b for b in preferred if b in all_bids] + [
    b for b in all_bids if b not in preferred
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

if in_future_region and fut_rul_pred.size > 0:
    idx_pred = int(np.argmin(np.abs(fut_cycles - current_cycle)))
    current_pred_rul = float(fut_rul_pred[idx_pred])
else:
    current_pred_rul = current_true_rul

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
remaining_days = current_pred_rul / max(cycles_per_day, 1e-6)
remaining_km = current_pred_rul * km_per_cycle

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

    # 🔹 상태 계산 (공통) – 여기서 한 번만 계산해서 카드 / GIF 둘 다 사용
    if initial_rul > 0 and not math.isnan(current_pred_rul):
        rul_pct = max(0.0, min(100.0, 100.0 * current_pred_rul / initial_rul))

        if rul_pct >= 60.0:
            status_txt, status_emoji, status_color = "양호", "🟢", "#2ca02c"
        elif rul_pct >= 30.0:
            status_txt, status_emoji, status_color = "주의", "🟡", "#ff7f0e"
        else:
            status_txt, status_emoji, status_color = "교체/정비 고려", "🔴", "#d62728"
    else:
        rul_pct = float("nan")
        status_txt, status_emoji, status_color = "N/A", "⚪", "#999999"

    # 🔹 3개 카드: 현재 사이클 / 예측 잔여 수명 / 현재 상태(작은 카드)
    h1, h2, h3 = st.columns(3)

    with h1:
        st.metric("현재 사이클", current_cycle)

    with h2:
        if not math.isnan(current_pred_rul):
            st.metric("예측 잔여 수명 (사이클)", f"{current_pred_rul:.1f}")
        else:
            st.metric("예측 잔여 수명 (사이클)", "N/A")

    with h3:
        # st.metric 대신 작은 커스텀 카드 (글자 확 줄임)
        if not math.isnan(rul_pct):
            st.markdown(
                f"""
                <div style='border-radius:10px; border:1px solid #e0e0e0;
                            padding:6px 10px; text-align:center;'>
                    <div style='font-size:11px; color:#777; margin-bottom:2px;'>현재 상태</div>
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
                    <div style='font-size:14px; font-weight:600; color:#999;'>N/A</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # 🔹 관측 전체 수명은 각주로만 노출 (너 보려고만)
    if not math.isnan(total_cycle_life):
        st.markdown(
            f"""
            <div style='font-size:11px; color:#999; margin-top:2px;'>
                <sup>1</sup> 관측 전체 수명 (사이클): {total_cycle_life:.0f}
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 배터리 요약 정보 (용량 중심)
    info_parts = []
    if not math.isnan(cap_init):
        info_parts.append(f"초기 용량: {cap_init:.3f} Ah")
    if not math.isnan(current_cap):
        info_parts.append(f"{current_cycle} 사이클 시 용량: {current_cap:.3f} Ah")
    if not math.isnan(cap_final):
        info_parts.append(f"최종 용량: {cap_final:.3f} Ah")

    if info_parts:
        st.markdown(
            "<div style='font-size:13px; color:#555; margin-bottom:6px;'>"
            + " · ".join(info_parts)
            + "</div>",
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
        st.image(str(gif_to_show), use_container_width=True)
        st.markdown(
            "<div style='font-size:11px; color:#888; text-align:center; margin-top:4px;'>"
            "위 GIF는 이해를 돕기 위한 예시 이미지입니다."
            "</div>",
            unsafe_allow_html=True,
        )


    # # 🔹 관측 전체 수명은 각주로만 표시 (너 보려고만)
    # if not math.isnan(total_cycle_life):
    #     st.caption(f"※ 관측 전체 수명 (사이클): {total_cycle_life:.0f}")

    # 배터리 요약 정보 (용량 중심)
    info_parts = []
    if not math.isnan(cap_init):
        info_parts.append(f"초기 용량: {cap_init:.3f} Ah")
    if not math.isnan(current_cap):
        info_parts.append(f"{current_cycle} 사이클 시 용량: {current_cap:.3f} Ah")
    if not math.isnan(cap_final):
        info_parts.append(f"최종 용량: {cap_final:.3f} Ah")

    if info_parts:
        st.markdown(
            "<div style='font-size:13px; color:#555; margin-bottom:6px;'>"
            + " · ".join(info_parts)
            + "</div>",
            unsafe_allow_html=True,
        )

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

                        s_col_a, s_col_b = st.columns(2)

                        # Baseline (A) 박스
                        with s_col_a:
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
                        with s_col_b:
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
                        slope = float(stats.get("slope_rul_per_unit", 0.0))
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
                            f"<div style='font-size:11px; color:#555;'>※ {guide}</div>",
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
        initial_rul_s = float(scenario_inputs["initial_rul"])
        current_cycle_s = int(scenario_inputs["current_cycle"])
        selected_bid_s = scenario_inputs["battery_id"]

        # 1) 시나리오 결과 계산 (선형 민감도 기반 RUL 보정)
        delta_rul = 0.0
        per_feat_details = []

        for feat_key in SCENARIO_FEATURES.keys():
            base_v = float(baseline[feat_key])
            scen_v = float(scenario[feat_key])
            delta_v = scen_v - base_v

            stats = FEATURE_STATS.get(feat_key, {})
            slope = float(stats.get("slope_rul_per_unit", 0.0))  # Δx 1 단위당 RUL 기울기
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
# Auto-play 루프
# =================================================
if st.session_state.get("auto_play", False):
    cur = st.session_state.get("play_cycle", current_cycle)
    if cur < max_cycle:
        # 한 번에 AUTO_PLAY_STEP 만큼 이동
        next_cycle = min(max_cycle, cur + AUTO_PLAY_STEP)
        st.session_state["play_cycle"] = next_cycle

        # 프레임 간 대기 시간
        time.sleep(AUTO_PLAY_DELAY_SEC)

        # 전체 rerun (Streamlit 구조상 어쩔 수 없이 전체 재실행)
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
