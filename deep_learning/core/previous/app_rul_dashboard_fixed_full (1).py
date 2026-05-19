import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# 경로 / 패키지 설정 (어디서 실행해도 deep_learning 패키지 잡히도록)
# -----------------------------------------------------------------------------
_THIS_FILE = Path(__file__).resolve()
_BACKEND_ROOT = None
for cand in _THIS_FILE.parents:
    # backend 루트 후보: 바로 아래에 deep_learning 디렉터리가 있는 디렉터리
    if (cand / "deep_learning").is_dir():
        _BACKEND_ROOT = cand
        break

if _BACKEND_ROOT is not None and str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

FILE_DIR = Path(__file__).resolve().parent          # .../backend/deep_learning/core
PROJECT_ROOT = FILE_DIR.parent.parent               # .../backend

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ASSETS_DIR = FILE_DIR / "assets"
LOADING_GIF = ASSETS_DIR / "loading.gif"
HEALTH_HIGH_GIF = ASSETS_DIR / "high.gif"
HEALTH_MED_GIF = ASSETS_DIR / "medium.gif"
HEALTH_LOW_GIF = ASSETS_DIR / "low.gif"

BMAML_DIR = FILE_DIR / "dashboard_export" / "bmaml"
CKPT_DEFAULT = FILE_DIR / "core_checkpoints" / "nasa_bmaml_best_re.pt"
SHAP_JSON = FILE_DIR / "shap_outputs" / "bmaml_shap_seq_feature_importance.json"

FEATURE_STATS_PATH = FILE_DIR / "analysis" / "feature_rul_stats.json"
NASA_FEATURES_PATH = FILE_DIR / "analysis" / "nasa_features_rul.csv"

EXPORT_TEST_BATTERIES = {"B0018", "B0033", "B0043", "B0055"}  # 첫 화면용
DEFAULT_R_RATIO = 0.25
DEFAULT_CYCLES_PER_DAY = 0.5  # 기본: 하루 0.5 싸이클 정도

MODEL_TAG = "Few-shot physics-informed meta learning RUL, transformer backbone)"

# 🔴 반드시 첫 Streamlit 호출이어야 하는 부분
st.set_page_config(
    page_title="Battery RUL Meta-Learning Dashboard (Real-time Inference)",
    layout="wide",
)
st.title("👁️🔋 실시간 리튬이온 배터리 RUL 예측 대시보드")
st.caption(MODEL_TAG)

# -----------------------------------------------------------------------------
# 내부 모듈 import (이제 경로 문제 없이 동작해야 함)
# -----------------------------------------------------------------------------
from deep_learning.core.prefix_inference_viz_meta_restored import (  # type: ignore
    build_model_and_grouped,
    make_task_prefix,
    run_adapt_and_predict,
)

from deep_learning.core.bmaml_runtime import (  # type: ignore
    build_runtime_records,
    build_runtimes_for_task,
)

from deep_learning.core.rul_precomputed_loader_restored import PrecomputedRULLoader  # type: ignore


# -----------------------------------------------------------------------------
# Precomputed Runtime Loader
# -----------------------------------------------------------------------------
def load_precomputed_index(folder: Path):
    index_path = folder / "index.json"
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def list_precomputed_r_ratios(index: Dict, battery_id: str) -> List[float]:
    if battery_id not in index:
        return []
    ratio_dict = index[battery_id]
    ratios = []
    for k in ratio_dict.keys():
        try:
            ratios.append(float(k))
        except ValueError:
            continue
    return sorted(ratios)


def find_best_r_ratio(index: Dict, battery_id: str, target: float) -> float:
    ratios = list_precomputed_r_ratios(index, battery_id)
    if not ratios:
        return float(DEFAULT_R_RATIO)
    best = min(ratios, key=lambda x: abs(x - target))
    return float(best)


def load_precomputed_runtime(precomputed_dir: Path):
    return PrecomputedRuntime(precomputed_root=precomputed_dir, default_r_ratio=DEFAULT_R_RATIO)


class PrecomputedRuntime:
    """
    간단한 in-memory precomputed runtime 관리 클래스.
    """

    def __init__(self, precomputed_root: Path, default_r_ratio: float = 0.25):
        self.root = precomputed_root
        self.default_r_ratio = float(default_r_ratio)
        self.index = load_precomputed_index(precomputed_root)
        self.loader = PrecomputedRULLoader(root=precomputed_root)

    def get_available_batteries(self) -> List[str]:
        return sorted(list(self.index.keys()))

    def get_available_r_ratios(self, battery_id: str) -> List[float]:
        return list_precomputed_r_ratios(self.index, battery_id)

    def get_closest_r_ratio(self, battery_id: str, target: float | None = None) -> float:
        if target is None:
            target = self.default_r_ratio
        return find_best_r_ratio(self.index, battery_id, float(target))

    def load_runtime(self, battery_id: str, r_ratio: float | None = None):
        """
        배터리 / r_ratio에 해당하는 precomputed runtime 반환
        """
        if r_ratio is None:
            r_ratio = self.default_r_ratio

        best_r = self.get_closest_r_ratio(battery_id, r_ratio)
        export_dict = self.loader.load(self.root, best_r)

        task_key = None
        for k in export_dict.keys():
            if k.startswith(str(battery_id)):
                task_key = k
                break

        if task_key is None:
            raise KeyError(f"[PrecomputedRuntime] battery_id={battery_id} 에 해당하는 키를 찾지 못했습니다.")

        return export_dict[task_key], best_r


# -----------------------------------------------------------------------------
# SHAP / NASA feature stats
# -----------------------------------------------------------------------------
def _load_shap_feature_importance(shap_json_path: Path) -> pd.DataFrame:
    if not shap_json_path.exists():
        return pd.DataFrame()

    with open(shap_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    feature_names = data.get("feature_names", [])
    shap_values = data.get("shap_values", [])

    if not feature_names or not shap_values:
        return pd.DataFrame()

    arr = np.asarray(shap_values, dtype=float)
    if arr.ndim == 2:
        mean_abs = np.mean(np.abs(arr), axis=0)
    elif arr.ndim == 3:
        mean_abs = np.mean(np.abs(arr), axis=(0, 1))
    else:
        arr = arr.reshape(-1, arr.shape[-1])
        mean_abs = np.mean(np.abs(arr), axis=0)

    df = pd.DataFrame({"feature": feature_names, "importance": mean_abs})
    df = df.sort_values("importance", ascending=False).reset_index(drop=True)
    return df


def _load_nasa_feature_stats(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df


def _human_readable_battery_id(bid: str) -> str:
    return f"Battery {bid}"


def _human_readable_r_ratio(r: float) -> str:
    return f"{r:.2f}"


@st.cache_resource(show_spinner=False)
def get_precomputed_runtime(root: Path) -> PrecomputedRuntime:
    return load_precomputed_runtime(root)


@st.cache_data(show_spinner=False)
def get_shap_importance(shap_json_path: Path) -> pd.DataFrame:
    return _load_shap_feature_importance(shap_json_path)


@st.cache_data(show_spinner=False)
def get_nasa_features(path: Path) -> pd.DataFrame:
    return _load_nasa_feature_stats(path)


# -----------------------------------------------------------------------------
# RUL / 남은 수명 계산 관련 유틸
# -----------------------------------------------------------------------------
DESIGN_EFC_EOL = 1000.0  # 설계상 1000 EFC에서 SoH ~80% 도달 가정


def _estimate_cycles_to_ratio_from_rul(rul_cycles: float, r_ratio: float) -> float:
    """
    간단 선형 근사: r_ratio = (cycles_used / DESIGN_EFC_EOL)
    => cycles_used = r_ratio * DESIGN_EFC_EOL
    => cycles_left = rul_cycles
    => ratio_now ~ (DESIGN_EFC_EOL - cycles_left) / DESIGN_EFC_EOL

    여기서는 display용으로만 사용 (rough indicator)
    """
    if not math.isfinite(rul_cycles):
        return float("nan")
    # 단순히 사용 비율 ~ (EOL - RUL) / EOL 로 보는 버전
    used = max(0.0, DESIGN_EFC_EOL - rul_cycles)
    ratio_now = used / max(1e-6, DESIGN_EFC_EOL)
    return float(ratio_now)


def _estimate_days_left(rul_cycles: float, cycles_per_day: float) -> float:
    if not math.isfinite(rul_cycles) or cycles_per_day <= 0:
        return float("nan")
    return float(rul_cycles / cycles_per_day)


# -----------------------------------------------------------------------------
# 시나리오 컨텍스트 (프론트에서 시나리오 여러 개 저장용)
# -----------------------------------------------------------------------------
class ScenarioContext:
    def __init__(self):
        # 각 scenario는 dict 로 관리:
        # {
        #   "name": str,
        #   "battery_id": str,
        #   "rul_cycles": float,
        #   "days_left": float,
        #   "r_ratio_display": float,
        # }
        self.scenarios: List[Dict] = []

    def add_scenario(self, info: Dict):
        self.scenarios.append(info)

    def to_table(self) -> pd.DataFrame:
        if not self.scenarios:
            return pd.DataFrame()
        return pd.DataFrame(self.scenarios)


# -----------------------------------------------------------------------------
# Streamlit session_state 초기화
# -----------------------------------------------------------------------------
if "scenarios" not in st.session_state:
    st.session_state["scenarios"] = ScenarioContext()

if "records" not in st.session_state:
    st.session_state["records"] = None  # API와 연동될 runtime records

if "meta_state" not in st.session_state:
    st.session_state["meta_state"] = None

if "bmaml_grouped" not in st.session_state:
    st.session_state["bmaml_grouped"] = None


# -----------------------------------------------------------------------------
# Meta state / runtime records 로딩 (한 번만)
# -----------------------------------------------------------------------------
def _ensure_meta_and_records():
    """
    meta_state, bmaml_grouped, runtime records를 lazy하게 한 번만 로드.
    (API에서 쓸 runtime records까지 같이 구성)
    """
    if st.session_state["records"] is not None and st.session_state["meta_state"] is not None:
        return

    with st.spinner("meta-learner & precomputed runtime 로딩 중..."):
        # (1) meta_state & grouped
        meta_state = build_model_and_grouped(str(CKPT_DEFAULT), eval_dataset="from_ckpt")
        st.session_state["meta_state"] = meta_state

        # (2) runtime records (API 연동용)
        try:
            records_rt = build_runtime_records(meta_state, EXPORT_TEST_BATTERIES)
            st.session_state["records"] = records_rt
        except Exception as e:
            st.error(f"runtime records 생성 실패: {e}")
            st.stop()


# -----------------------------------------------------------------------------
# UI 레이아웃: 사이드바 (배터리 선택 / r_ratio / cycles_per_day)
# -----------------------------------------------------------------------------
st.sidebar.header("⚙️ 실험 설정")

repo = get_precomputed_runtime(BMAML_DIR)
available_batteries = repo.get_available_batteries()

# 첫 화면에는 EXPORT_TEST_BATTERIES 우선
preferred = [b for b in EXPORT_TEST_BATTERIES if b in available_batteries]
others = [b for b in available_batteries if b not in preferred]
battery_choices = preferred + others

if not battery_choices:
    st.sidebar.error("사전 계산된 배터리 결과가 없습니다. (bmaml/index.json 확인 필요)")
    st.stop()

default_battery = preferred[0] if preferred else battery_choices[0]

battery_id = st.sidebar.selectbox(
    "🔋 NASA 배터리 선택 (실험 배터리)",
    battery_choices,
    index=battery_choices.index(default_battery),
    format_func=_human_readable_battery_id,
)

available_r_list = repo.get_available_r_ratios(battery_id)
if not available_r_list:
    st.sidebar.warning("이 배터리에 대해 precomputed r_ratio 정보가 없습니다. 기본값을 사용합니다.")
    default_r_ratio = DEFAULT_R_RATIO
else:
    default_r_ratio = available_r_list[0]

user_r_ratio = st.sidebar.slider(
    "fine-tuning prefix 비율 (r_ratio)",
    min_value=0.05,
    max_value=0.5,
    value=float(default_r_ratio),
    step=0.05,
)

cycles_per_day = st.sidebar.slider(
    "평균 하루 싸이클 수 (usage intensity)",
    min_value=0.1,
    max_value=2.0,
    value=float(DEFAULT_CYCLES_PER_DAY),
    step=0.1,
)

st.sidebar.markdown("---")
scenario_name = st.sidebar.text_input("시나리오 이름 (optional)", value="")
add_scenario_clicked = st.sidebar.button("현재 설정으로 시나리오 추가")


# -----------------------------------------------------------------------------
# meta-state / runtime records 준비
# -----------------------------------------------------------------------------
_ensure_meta_and_records()  # lazy load

meta_state = st.session_state["meta_state"]
records_rt = st.session_state["records"]

if meta_state is None or records_rt is None:
    st.error("meta_state 또는 runtime records가 초기화되지 않았습니다.")
    st.stop()

# -----------------------------------------------------------------------------
# 메인 컬럼 레이아웃
# -----------------------------------------------------------------------------
left_col, right_col = st.columns([1.2, 1.0])

# -----------------------------------------------------------------------------
# 왼쪽: 선택된 배터리 / r_ratio에 대한 RUL 예측
# -----------------------------------------------------------------------------
with left_col:
    st.subheader("📈 선택 배터리 · meta-learned RUL inference")

    try:
        # 사전 계산 runtime 로드 (prefix / s_post 등 precomputed export)
        precomputed_runtime, used_r_ratio = repo.load_runtime(battery_id, user_r_ratio)

        # run_adapt_and_predict 로 실제 RUL 예측 수행
        with st.spinner("meta-learned 모델이 prefix를 활용해 RUL 추정 중..."):
            # precomputed_runtime 안에 prefix-like structure가 들어있다고 가정
            # (bmaml_runtime.build_runtimes_for_task 와 호환되게 설계됨)
            task_prefix = make_task_prefix(precomputed_runtime)
            rul_result = run_adapt_and_predict(meta_state, task_prefix)

        rul_cycles = float(rul_result.get("pred_rul_cycles", float("nan")))
        rul_std = float(rul_result.get("pred_rul_std", float("nan")))

        ratio_used = _estimate_cycles_to_ratio_from_rul(rul_cycles, used_r_ratio)
        days_left = _estimate_days_left(rul_cycles, cycles_per_day)

        # 상단 카드 3개 정도로 요약
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(
                "예측 잔여 싸이클 (RUL)",
                f"{rul_cycles:,.0f} cycles" if math.isfinite(rul_cycles) else "N/A",
                help="meta-learned BMAML 모델이 예측한 남은 싸이클 수",
            )
        with m2:
            st.metric(
                "예측 일수 (rough)",
                f"{days_left:,.0f} days" if math.isfinite(days_left) else "N/A",
                help=f"하루 {cycles_per_day:.2f} cycles 기준, 단순 환산",
            )
        with m3:
            st.metric(
                "사용 비율 (rough)",
                f"{ratio_used*100:,.1f} %" if math.isfinite(ratio_used) else "N/A",
                help="RUL 기반으로 추산한 사용 비율 (rough indicator)",
            )

        st.caption(
            f"· precomputed r_ratio={used_r_ratio:.2f} 기준 runtime 사용\n"
            f"· CKPT: {CKPT_DEFAULT.name}"
        )

    except Exception as e:
        st.error(f"RUL 예측 과정에서 오류가 발생했습니다: {e}")
        rul_cycles = float("nan")
        days_left = float("nan")
        ratio_used = float("nan")

    # 시나리오 저장 버튼
    if add_scenario_clicked and math.isfinite(rul_cycles):
        sc_name = scenario_name.strip() or f"{battery_id} (r={user_r_ratio:.2f})"
        st.session_state["scenarios"].add_scenario(
            {
                "name": sc_name,
                "battery_id": battery_id,
                "r_ratio": float(user_r_ratio),
                "rul_cycles": rul_cycles,
                "days_left": days_left,
                "usage_ratio": ratio_used,
            }
        )
        st.success(f"시나리오 '{sc_name}'가 저장되었습니다.")


# -----------------------------------------------------------------------------
# 오른쪽: 헬스 게이지 / GIF / feature importance
# -----------------------------------------------------------------------------
with right_col:
    st.subheader("🧠 배터리 건강도 상태 요약")

    # 헬스 상태에 따라 GIF 선택
    if math.isfinite(ratio_used):
        if ratio_used < 0.4:
            gif_to_show = HEALTH_HIGH_GIF
            health_label = "건강 양호"
        elif ratio_used < 0.7:
            gif_to_show = HEALTH_MED_GIF
            health_label = "중간 수준"
        else:
            gif_to_show = HEALTH_LOW_GIF
            health_label = "노화 진행"
    else:
        gif_to_show = LOADING_GIF
        health_label = "계산 불가"

    st.markdown(f"**현재 추정 상태: {health_label}**")
    if gif_to_show.exists():
        st.image(str(gif_to_show))
    else:
        st.write("(health GIF를 찾을 수 없습니다.)")

    st.markdown("---")

    # SHAP 기반 feature importance 요약
    shap_df = get_shap_importance(SHAP_JSON)
    if shap_df.empty:
        st.info("SHAP feature importance 정보를 찾을 수 없습니다.")
    else:
        st.markdown("#### 🔍 BMAML 시퀀스 피처 중요도 (SHAP 기반)")
        top_k = 10
        top_df = shap_df.head(top_k)
        fig = go.Figure(
            data=go.Bar(
                x=top_df["importance"][::-1],
                y=top_df["feature"][::-1],
                orientation="h",
            )
        )
        fig.update_layout(
            height=320,
            margin=dict(l=10, r=10, t=30, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# 아래: 저장된 시나리오 테이블 / NASA feature stats
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("📚 시나리오 비교 & NASA feature 통계")

sc_df = st.session_state["scenarios"].to_table()
if sc_df.empty:
    st.info("저장된 시나리오가 없습니다. 사이드바에서 '현재 설정으로 시나리오 추가'를 눌러보세요.")
else:
    st.markdown("#### 저장된 시나리오 목록")
    st.dataframe(sc_df, use_container_width=True)

nasa_df = get_nasa_features(NASA_FEATURES_PATH)
if nasa_df.empty:
    st.info("NASA feature 통계 CSV를 찾을 수 없습니다.")
else:
    st.markdown("#### NASA 셀 데이터 기반 feature-RUL 통계 (샘플)")
    st.dataframe(nasa_df.head(20), use_container_width=True)

# -----------------------------------------------------------------------------
# 하단 설명
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    ### ℹ️ 설명

    - 이 대시보드는 NASA 배터리 싸이클 데이터셋을 기반으로 학습된  
      **BMAML (meta-learning) RUL 모델**의 예측 결과를 시각적으로 보여줍니다.
    - `r_ratio`는 prefix로 사용하는 과거 싸이클 비율로,  
      값이 클수록 더 많은 이력을 기반으로 적응(fine-tuning)합니다.
    - 하루 평균 싸이클 수(`cycles_per_day`)를 사용해서  
      예측된 잔여 싸이클(RUL)을 rough하게 일수로 환산합니다.
    - 오른쪽 패널의 health GIF는 예측된 사용 비율(rough usage ratio)에 따라  
      **양호 / 중간 / 노화 진행** 상태를 직관적으로 표현합니다.
    """
)
