#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
app_rul_with_ev6.py

1) NASA 셀 테스트 기반 BMAML RUL 대시보드
2) Kia EV6 실제 주행 로그 기반 EFC/SoH RUL 대시보드

를 하나의 Streamlit 앱에서 탭으로 전환하며 볼 수 있게 한 통합 스크립트.

실행 예시
---------
$ cd /Users/velocitygoal/Desktop/battery_project/v11
$ streamlit run deep_learning/core/app_rul_with_ev6.py
"""

import glob
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -------------------------------------------------
# 공통 Paths / sys.path 설정
# -------------------------------------------------
FILE_DIR = Path(__file__).resolve().parent  # .../v11/deep_learning/core
PROJECT_ROOT = FILE_DIR.parent.parent       # .../v11

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ASSETS_DIR = FILE_DIR / "assets"
LOADING_GIF = ASSETS_DIR / "loading.gif"
HEALTH_HIGH_GIF = ASSETS_DIR / "high.gif"
HEALTH_MED_GIF = ASSETS_DIR / "medium.gif"
HEALTH_LOW_GIF = ASSETS_DIR / "low.gif"

# NASA BMAML 관련
BMAML_DIR = FILE_DIR / "dashboard_export" / "bmaml"
CKPT_DEFAULT = FILE_DIR / "core_checkpoints" / "nasa_bmaml_best_re.pt"
SHAP_JSON = FILE_DIR / "shap_outputs" / "bmaml_shap_seq_feature_importance.json"
FEATURE_STATS_PATH = FILE_DIR / "analysis" / "feature_rul_stats.json"
EXPORT_TEST_BATTERIES = {"B0018", "B0033", "B0042", "B0043"}
DEFAULT_R_RATIO = 0.25

# EV6 관련
REAL_TIME_DIR = FILE_DIR / "real_time"
EV6_FULL_PATH = REAL_TIME_DIR / "ev6_with_efc_rul.csv"
EV6_DAILY_PATH = REAL_TIME_DIR / "ev6_daily_summary.csv"
USABLE_KWH = 74.6        # EV6 usable capacity (rough)
DESIGN_EFC_EOL = 1000.0  # 1000 EFC에서 SoH 80% 근사
SOH_EOL = 80.0

# BMAML core
from deep_learning.core.prefix_inference_viz_meta_restored_v3_pyc import (
    build_model_and_grouped,
    make_task_prefix,
    run_adapt_and_predict,
)

# -------------------------------------------------
# 시나리오 빌더용 Feature 정의 (NASA 쪽)
# -------------------------------------------------
SCENARIO_FEATURES = {
    "soh": {
        "label": "배터리 건강도 SoH (0~1)",
        "fallback_min": 0.6,
        "fallback_max": 1.0,
        "step": 0.01,
    },
    "regen_strength": {
        "label": "용량 저하량 (regen_strength)",
        "fallback_min": 0.0,
        "fallback_max": 5.0,
        "step": 0.1,
    },
    "voltage_min": {
        "label": "최소 전압 (V)",
        "fallback_min": 3.0,
        "fallback_max": 4.2,
        "step": 0.01,
    },
}

# -------------------------------------------------
# Small CSS
# -------------------------------------------------
st.set_page_config(
    page_title="Battery RUL Dashboard (NASA + EV6)",
    layout="wide",
)

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
# NASA: Data loaders / utils
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
def load_feature_stats(path: Path):
    """export_nasa_feature_rul_stats.py가 만든 feature_rul_stats.json 로드."""
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


FEATURE_STATS = load_feature_stats(FEATURE_STATS_PATH)
shap_names, shap_vals = load_shap_importance(SHAP_JSON)


@st.cache_resource(show_spinner="BMAML meta-learner & cycle DB 로딩 중...")
def load_meta_state(ckpt_path: str, eval_dataset: str = "from_ckpt"):
    return build_model_and_grouped(ckpt_path, eval_dataset=eval_dataset)


def build_runtime_records(meta_state, r_ratio: float = 0.3) -> Dict[str, dict]:
    """
    BMAML 실시간 추론으로 dashboard용 records 생성.
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

        cyc_arr = np.asarray(g.get("cycle", []), dtype=float)
        cycle_life_obs = float(cyc_arr.max()) if cyc_arr.size > 0 else float("nan")

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

        records[bid] = rec

    # 두 번째 추론에서는 33번 배터리는 제외 (BMAML eval 셋 기준)
    records.pop("B0033", None)
    return records


def run_bmaml_once(r_ratio: float, note: str = ""):
    """
    버튼 클릭 시 호출:
    - BMAML meta-state 로드 (cache_resource)
    - runtime records 생성 후 session_state에 저장
    """
    loading_placeholder = st.empty()
    with loading_placeholder.container():
        st.markdown(
            f"""
            ### 🔧 BMAML meta-learner 초기화 중...
            - 체크포인트 로드
            - NASA 메타 DB에서 테스트 배터리 셀 로드
            - r_ratio = {r_ratio:.2f} 기준 few-shot meta-adaptation 수행

            예측은 GPU/CPU 환경에 따라 시간이 걸릴 수 있습니다.
            로딩 중에는 화면을 가능한 한 건드리지 않는 것을 권장합니다.

            {note}
            """
        )
        if LOADING_GIF.exists():
            st.image(str(LOADING_GIF), use_container_width=True)

    try:
        with st.spinner("BMAML meta-learner & battery tasks 준비 중..."):
            meta_state = load_meta_state(str(CKPT_DEFAULT), eval_dataset="from_ckpt")
            records_rt = build_runtime_records(meta_state=meta_state, r_ratio=r_ratio)

        st.session_state["records"] = records_rt
        st.session_state["r_ratio"] = float(r_ratio)
        st.session_state["records_source"] = f"runtime BMAML (r_ratio={r_ratio:.2f})"
        loading_placeholder.empty()

    except Exception as e:
        loading_placeholder.empty()
        st.error(
            "BMAML runtime records 생성 실패.\n"
            f"- error: {e}"
        )
        st.stop()


# -------------------------------------------------
# EV6: Data loaders / utils
# -------------------------------------------------
@st.cache_data(show_spinner="EV6 full CSV 로딩 중...")
def load_ev6_full(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    df = pd.read_csv(path)
    if "TimeStamp" in df.columns:
        df["TimeStamp"] = pd.to_datetime(df["TimeStamp"], errors="coerce", utc=True)
        df = df.sort_values("TimeStamp").reset_index(drop=True)
    return df


@st.cache_data(show_spinner="EV6 일별 요약 CSV 로딩 중...")
def load_ev6_daily(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(str(path))
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def compute_ev6_global_km_per_cycle(df: pd.DataFrame):
    """전체 EV6 데이터 기준: 총 거리(km), 총 EFC → km/EFC."""
    dist_km = np.nan
    if "DistanceTotal" in df.columns:
        dist_m = df["DistanceTotal"].max() - df["DistanceTotal"].min()
        dist_km = float(dist_m) / 1000.0
    elif "distance_km" in df.columns:
        dist_km = float(df["distance_km"].sum())

    if "EFC" in df.columns:
        efc_vals = df["EFC"].to_numpy(dtype=float)
        mask = np.isfinite(efc_vals)
        if mask.any():
            total_efc = float(np.nanmax(efc_vals[mask]) - np.nanmin(efc_vals[mask]))
        else:
            total_efc = np.nan
    else:
        total_efc = np.nan

    if dist_km and total_efc and total_efc > 0:
        km_per_cycle = dist_km / total_efc
    else:
        km_per_cycle = np.nan

    return dist_km, km_per_cycle


# -------------------------------------------------
# 시나리오 메세지 state
# -------------------------------------------------
if "scenarios" not in st.session_state:
    st.session_state["scenarios"] = []
if "scenario_message" not in st.session_state:
    st.session_state["scenario_message"] = None

# -------------------------------------------------
# 메인 탭
# -------------------------------------------------
st.title("🔋 Battery RUL Dashboard")
st.caption("NASA cell BMAML RUL + Kia EV6 real-driving EFC/SoH RUL")

tab_nasa, tab_ev6 = st.tabs(
    [
        "🧪 NASA BMAML RUL (셀 시험)",
        "🚗 EV6 Real Driving RUL (CAN 로그)",
    ]
)

# =================================================
# TAB 1: NASA BMAML RUL 대시보드
# =================================================
with tab_nasa:
    MODEL_TAG = "few-shot BMAML-SVGD (physics-informed meta RUL, CEEMDAN–Transformer–DNN backbone)"
    st.subheader("🧪 NASA Cell-based BMAML RUL Dashboard")
    st.caption(MODEL_TAG)

    # 0. records 초기화 (precomputed)
    if "records" not in st.session_state:
        precomputed = filter_export_demo(load_battery_records(BMAML_DIR))
        if precomputed:
            st.session_state["records"] = precomputed
            st.session_state["r_ratio"] = DEFAULT_R_RATIO
            st.session_state["records_source"] = "precomputed exports (r_ratio≈0.30)"

    records = st.session_state.get("records", {})
    current_r_ratio = st.session_state.get("r_ratio", DEFAULT_R_RATIO)
    records_source = st.session_state.get("records_source", "precomputed exports")

    # Sidebar: 1) 모델 설정
    st.sidebar.header("NASA BMAML 설정")
    if not CKPT_DEFAULT.exists():
        st.sidebar.error(f"BMAML checkpoint not found:\n{CKPT_DEFAULT}")
        st.stop()

    r_ratio = st.sidebar.slider(
        "초기 적응 비율 (r_ratio)",
        min_value=0.1,
        max_value=0.8,
        value=float(current_r_ratio),
        step=0.05,
        help="처음 보는 배터리 사이클 중 앞부분 몇 %를 '초기 학습용 데이터'로 사용할지 설정합니다.",
    )
    init_clicked = st.sidebar.button("모델 초기화 & 재적응 실행")

    if init_clicked:
        run_bmaml_once(r_ratio=r_ratio)

    # 아직 records가 없으면 안내
    if "records" not in st.session_state or not st.session_state["records"]:
        st.info(
            "왼쪽 사이드바에서 **모델 초기화 & 재적응 실행** 버튼을 눌러 주세요.\n\n"
            "예측/추론은 약 1분 내외로 소요됩니다."
        )
        st.stop()

    records = st.session_state["records"]
    current_r_ratio = st.session_state.get("r_ratio", r_ratio)
    records_source = st.session_state.get("records_source", "precomputed exports")

    # Sidebar: 2) 배터리 & 사이클 선택
    st.sidebar.header("NASA 테스트 배터리 선택")

    all_bids = sorted(records.keys())
    preferred = ["B0018", "B0033", "B0042", "B0043"]
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

    # 사이클 자동재생
    auto_play = st.sidebar.checkbox(
        "사이클 자동 재생 (NASA 탭)", key="auto_play_nasa", value=False
    )

    current_cycle_slider = st.sidebar.slider(
        "현재 사이클 (수동 모드)",
        min_value=min_cycle,
        max_value=max_cycle,
        step=1,
        value=st.session_state.get("play_cycle", max(min_cycle, 1)),
        key="current_cycle_slider_nasa",
        disabled=st.session_state.get("auto_play_nasa", False),
        help="커서를 BMAML이 실제로 예측하는 구간(초기 r_ratio 이후)으로 옮기면, 예측 RUL 기반 상태를 볼 수 있습니다.",
    )

    if "play_cycle" not in st.session_state:
        st.session_state["play_cycle"] = current_cycle_slider

    if st.session_state.get("auto_play_nasa", False):
        current_cycle = int(st.session_state["play_cycle"])
    else:
        current_cycle = int(current_cycle_slider)
        st.session_state["play_cycle"] = current_cycle

    # Sidebar: BMAML metrics
    st.sidebar.markdown("---")
    rmse = rec.get("rmse", float("nan"))
    mae = rec.get("mae", float("nan"))

    st.sidebar.subheader("NASA RUL 예측 오차")
    if not math.isnan(rmse):
        st.sidebar.metric("RMSE (future region)", f"{rmse:.2f}")
    if not math.isnan(mae):
        st.sidebar.metric("MAE (future region)", f"{mae:.2f}")

    st.sidebar.markdown("---")
    st.sidebar.caption(
        f"Records source: **{records_source}** · r_ratio: {current_r_ratio:.2f}"
    )

    # 공통 값들 (cycles / RUL)
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

    cap_init = rec.get("cap_init", float("nan"))
    cap_final = rec.get("cap_final", float("nan"))
    cycle_life_obs = rec.get("cycle_life_obs", float("nan"))

    if has_capacity:
        current_cap = float(capacity_curve[idx_slider])
    else:
        current_cap = float("nan")

    # Usage & EOL 공유 값
    cycles_per_day = st.session_state.get("cycles_per_day", 1.0)
    eol_threshold = st.session_state.get("eol_threshold", 80)

    health_pct = max(
        0.0, min(100.0, 100.0 * current_pred_rul / max(initial_rul, 1e-6))
    )

    if health_pct >= eol_threshold:
        health_status, health_emoji, color = "양호", "🟢", "#2ca02c"
    elif health_pct >= 0.5 * eol_threshold:
        health_status, health_emoji, color = "점검 권장", "🟡", "#ff7f0e"
    else:
        health_status, health_emoji, color = "교체/정비 필요", "🔴", "#d62728"

    km_per_cycle_assumed = st.session_state.get("km_per_cycle", 400.0)
    remaining_days = current_pred_rul / max(cycles_per_day, 1e-6)
    remaining_km = current_pred_rul * km_per_cycle_assumed

    # 2x2 Layout
    top_left, top_right = st.columns([2.0, 2.0])
    bottom_left, bottom_right = st.columns([2.0, 2.0])

    # TOP-LEFT: info + GIF
    with top_left:
        st.markdown(
            f"<div style='font-size:22px; font-weight:700;'>NASA Cell · {selected_bid}</div>",
            unsafe_allow_html=True,
        )
        info_parts = []
        if not math.isnan(cap_init):
            info_parts.append(f"초기 용량: {cap_init:.3f} Ah")
        if not math.isnan(current_cap):
            info_parts.append(f"{current_cycle} cycle 시 용량: {current_cap:.3f} Ah")
        if not math.isnan(cap_final):
            info_parts.append(f"최종 용량: {cap_final:.3f} Ah")
        if not math.isnan(cycle_life_obs):
            info_parts.append(f"관측 수명: {cycle_life_obs:.0f} cycles")

        if info_parts:
            st.markdown(
                "<div style='font-size:13px; color:#555; margin-bottom:6px;'>"
                + " · ".join(info_parts)
                + "</div>",
                unsafe_allow_html=True,
            )

        h1, h2, h3 = st.columns(3)
        with h1:
            st.metric("현재 사이클", current_cycle)
        with h2:
            st.metric("예측 잔여 수명 (cycles)", f"{current_pred_rul:.1f}")
        with h3:
            st.markdown(
                f"<div style='font-size:16px; font-weight:600; color:{color};'>"
                f"{health_emoji} 상태: {health_status} · {health_pct:.1f}%</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div style='font-size:12px; color:#000; margin-top:4px;'>
                - 하루 {cycles_per_day:.1f} cycle 사용 가정 시, 남은 사용 기간: 약 {remaining_days:.1f}일<br/>
                - EV6 롱레인지 기준 (1 cycle ≈ {km_per_cycle_assumed:.0f} km) 환산 시<br/>
                &nbsp;&nbsp;→ 대략 <b>{remaining_km:,.0f} km</b> 정도 주행 가능 (scale 감각용).
            </div>
            """,
            unsafe_allow_html=True,
        )

        gif_to_show = None
        if health_pct >= eol_threshold and HEALTH_HIGH_GIF.exists():
            gif_to_show = HEALTH_HIGH_GIF
        elif health_pct >= 0.5 * eol_threshold and HEALTH_MED_GIF.exists():
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

    # TOP-RIGHT: RUL trajectory
    with top_right:
        st.markdown(f"### RUL 궤적 (임계치 80% 기준) · {selected_bid}")
        fig = go.Figure()

        # 실제 RUL 전체
        fig.add_trace(
            go.Scatter(
                x=cycles,
                y=rul_true,
                mode="lines",
                name="실제 RUL",
                line=dict(color="rgba(0,0,0,0.5)", dash="dash"),
            )
        )

        # 과거 관측 RUL
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

        # 미래 예측 + 불확실성
        if in_future_region and fut_cycles.size > 0:
            mask_future = fut_cycles >= float(current_cycle)
            fut_x = fut_cycles[mask_future]
            fut_pred_y = fut_rul_pred[mask_future]
            fut_std_y = fut_rul_std[mask_future]

            if fut_x.size > 0:
                if fut_std_y.size == fut_pred_y.size and not np.all(
                    np.isnan(fut_std_y)
                ):
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
                            name="예측 불확실성 (±2σ)",
                        )
                    )

                fig.add_trace(
                    go.Scatter(
                        x=fut_x,
                        y=fut_pred_y,
                        mode="lines",
                        name="예측 RUL (BMAML-SVGD)",
                        line=dict(color="rgb(214,39,40)"),
                    )
                )

        # 현재 사이클 vertical line
        fig.add_vline(
            x=float(current_cycle),
            line_width=2,
            line_dash="dot",
            line_color="black",
            annotation_text="현재 cycle",
            annotation_position="top",
        )

        fig.update_layout(
            xaxis_title="Cycle index",
            yaxis_title="RUL (cycles)",
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

        st.caption(
            "초기 r_ratio 구간은 BMAML의 meta-adaptation에 사용되고, "
            "그 이후 구간에서 잔여 수명 RUL 궤적을 예측합니다. "
            "r_ratio를 줄이면 더 적은 초기 데이터로 예측을 시도하는 셈입니다."
        )

    # BOTTOM-LEFT: Usage & EOL / Scenario input
    with bottom_left:
        st.markdown("### Usage & EOL / Scenario input (NASA)")
        cycles_per_day_sb = st.number_input(
            "평균 하루 주행 사이클 수",
            min_value=0.1,
            max_value=5.0,
            value=st.session_state.get("cycles_per_day", 1.0),
            step=0.1,
            key="cycles_per_day",
        )
        eol_threshold_sb = st.slider(
            "EOL 기준 (health %)",
            min_value=60,
            max_value=90,
            value=st.session_state.get("eol_threshold", 80),
            step=5,
            key="eol_threshold",
        )

        if FEATURE_STATS is None:
            st.warning(
                "feature_rul_stats.json을 찾을 수 없습니다. "
                "먼저 export_nasa_feature_rul_stats.py를 실행해 주세요."
            )
        else:
            st.markdown("#### Baseline vs Scenario (열화 조건 – NASA cell 통계)")
            baseline_vals = {}
            scenario_vals = {}

            for feat_key, meta in SCENARIO_FEATURES.items():
                stats = FEATURE_STATS.get(feat_key, {})
                lo = float(stats.get("q10", meta["fallback_min"]))
                hi = float(stats.get("q90", meta["fallback_max"]))
                mean_val = float(stats.get("mean", (lo + hi) / 2))

                col_a, col_b = st.columns(2)
                with col_a:
                    baseline_vals[feat_key] = st.slider(
                        meta["label"] + " · A (Baseline)",
                        min_value=lo,
                        max_value=hi,
                        value=mean_val,
                        step=meta["step"],
                    )
                with col_b:
                    scenario_vals[feat_key] = st.slider(
                        meta["label"] + " · B (Scenario)",
                        min_value=lo,
                        max_value=hi,
                        value=mean_val,
                        step=meta["step"],
                    )

            st.session_state["scenario_inputs"] = {
                "baseline": baseline_vals,
                "scenario": scenario_vals,
                "cycles_per_day": cycles_per_day_sb,
                "eol_threshold": eol_threshold_sb,
                "current_pred_rul": current_pred_rul,
                "current_true_rul": current_true_rul,
                "initial_rul": initial_rul,
                "current_cycle": current_cycle,
                "battery_id": selected_bid,
            }

    # BOTTOM-RIGHT: Scenario 계산 + 결과
    with bottom_right:
        st.markdown("### Scenario results (NASA cell 기반 what-if)")

        scenario_inputs = st.session_state.get("scenario_inputs", None)

        if scenario_inputs is None or FEATURE_STATS is None:
            st.info("왼쪽에서 baseline / scenario 조건을 먼저 설정해 주세요.")
        else:
            baseline = scenario_inputs["baseline"]
            scenario = scenario_inputs["scenario"]
            cycles_per_day_sb = scenario_inputs["cycles_per_day"]

            current_pred_rul_s = float(scenario_inputs["current_pred_rul"])
            current_true_rul_s = float(scenario_inputs["current_true_rul"])
            initial_rul_s = float(scenario_inputs["initial_rul"])
            current_cycle_s = int(scenario_inputs["current_cycle"])
            selected_bid_s = scenario_inputs["battery_id"]

            add_clicked = st.button("NASA Scenario RUL 계산 & 테이블 추가")

            if add_clicked:
                delta_rul = 0.0
                per_feat_details = []

                for feat_key in SCENARIO_FEATURES.keys():
                    base_v = float(baseline[feat_key])
                    scen_v = float(scenario[feat_key])
                    delta_v = scen_v - base_v

                    stats = FEATURE_STATS.get(feat_key, {})
                    slope = float(stats.get("slope_rul_per_unit", 0.0))

                    contrib = slope * delta_v
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

                scen_rul = max(0.0, current_pred_rul_s + delta_rul)
                scen_health_pct = max(
                    0.0, min(100.0, 100.0 * scen_rul / max(initial_rul_s, 1e-6))
                )

                remaining_days_base = current_pred_rul_s / max(
                    cycles_per_day_sb, 1e-6
                )
                remaining_days_scen = scen_rul / max(cycles_per_day_sb, 1e-6)

                scenario_row = {
                    "battery_id": selected_bid_s,
                    "cycle": current_cycle_s,
                    "rul_true": current_true_rul_s,
                    "rul_model_base": current_pred_rul_s,
                    "rul_model_scenario": scen_rul,
                    "health_pct_base": 100.0
                    * current_pred_rul_s
                    / max(initial_rul_s, 1e-6),
                    "health_pct_scenario": scen_health_pct,
                    "cycles_per_day": cycles_per_day_sb,
                    "remaining_days_base": remaining_days_base,
                    "remaining_days_scenario": remaining_days_scen,
                }
                for d in per_feat_details:
                    scenario_row[f"{d['feature']}_delta"] = d["delta_val"]

                st.session_state["scenarios"].append(scenario_row)

                diff_cycles = scen_rul - current_pred_rul_s
                if abs(diff_cycles) < 1e-3:
                    msg_text = (
                        "현재 설정은 baseline과 거의 동일합니다.\n\n"
                        "잔여 수명을 늘리고 싶다면:\n"
                        "- SoH(건강도)를 높이고\n"
                        "- 용량 저하량(regen_strength)을 줄이고\n"
                        "- 방전 시 최소 전압을 너무 낮게 쓰지 않는 운행 패턴이 유리한 경향이 있습니다.\n\n"
                        "(효과는 NASA cell 데이터 안에서의 **단순 선형 근사**입니다. "
                        "정밀한 물리 시뮬레이터라기보다는 방향성을 보는 용도입니다.)"
                    )
                    st.session_state["scenario_message"] = {
                        "type": "info",
                        "text": msg_text,
                    }
                else:
                    sign_word = "늘어납니다" if diff_cycles > 0 else "줄어듭니다"
                    msg_text = (
                        f"Scenario added: RUL {current_pred_rul_s:.1f} → {scen_rul:.1f} cycles "
                        f"({diff_cycles:+.1f} cycles, "
                        f"~{remaining_days_base:.1f} → ~{remaining_days_scen:.1f} days, {sign_word})."
                    )
                    st.session_state["scenario_message"] = {
                        "type": "success",
                        "text": msg_text,
                    }

                # 방금 계산된 메시지
                msg = st.session_state["scenario_message"]
                if msg["type"] == "info":
                    st.info(msg["text"])
                else:
                    st.success(msg["text"])

                st.markdown("#### 피처별 기여 (NASA cell 데이터 기반 근사)")

                for d in per_feat_details:
                    if abs(d["delta_val"]) < 1e-6:
                        continue
                    direction = "올리면" if d["delta_val"] > 0 else "내리면"
                    eff = "늘어나는" if d["delta_rul"] > 0 else "줄어드는"
                    st.markdown(
                        f"- **{d['feature']}** 값을 {direction}, "
                        f"RUL이 대략 {d['delta_rul']:+.1f} cycles만큼 {eff} 경향 "
                        f"(slope ≈ {d['slope_rul_per_unit']:.2f} cycles/unit)."
                    )

        # 마지막 메시지 항상 표시
        msg = st.session_state.get("scenario_message")
        if msg and not st.button("시나리오 메시지 새로고침", key="dummy_nasa"):
            if msg["type"] == "info":
                st.info(msg["text"])
            else:
                st.success(msg["text"])

        # 누적 시나리오 테이블 / CSV
        if st.session_state["scenarios"]:
            df_scen = pd.DataFrame(st.session_state["scenarios"])
            st.dataframe(df_scen, use_container_width=True, height=220)
            csv_bytes = df_scen.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download scenarios as CSV (NASA)",
                data=csv_bytes,
                file_name="nasa_rul_scenarios.csv",
                mime="text/csv",
            )

    # Auto-play 루프
    if st.session_state.get("auto_play_nasa", False):
        cur = st.session_state.get("play_cycle", current_cycle)
        if cur < max_cycle:
            st.session_state["play_cycle"] = cur + 1

# =================================================
# TAB 2: EV6 Real Driving RUL
# =================================================
with tab_ev6:
    st.subheader("🚗 Kia EV6 – Real Driving RUL Dashboard (준비중)")
    st.caption("EV6 CAN 로그 기반 EFC/SoH simple RUL 근사 (real_time.py output 사용)")

    # 파일 체크
    missing_full = not EV6_FULL_PATH.exists()
    missing_daily = not EV6_DAILY_PATH.exists()

    if missing_full or missing_daily:
        st.error(
            "EV6 전처리 CSV를 찾을 수 없습니다.\n\n"
            f"- full: `{EV6_FULL_PATH}`\n"
            f"- daily: `{EV6_DAILY_PATH}`\n\n"
            "터미널에서 먼저 아래 명령을 실행해 주세요:\n\n"
            "```bash\n"
            "cd /Users/velocitygoal/Desktop/battery_project/v11\n"
            "python -m deep_learning.core.real_time \\\n"
            "  --csv deep_learning/core/real_time/decoded_ev6_data_full.csv \\\n"
            "  --out deep_learning/core/real_time/ev6_daily_summary.csv\n"
            "```"
        )
        st.stop()

    df_full = load_ev6_full(EV6_FULL_PATH)
    df_daily = load_ev6_daily(EV6_DAILY_PATH)

    if "TimeStamp" in df_full.columns:
        df_full = df_full[df_full["TimeStamp"].notna()].reset_index(drop=True)

    if df_full.empty:
        st.error("EV6 full 데이터프레임이 비어 있습니다. real_time.py 전처리 결과를 확인해 주세요.")
        st.stop()

    total_km, km_per_cycle_ev6 = compute_ev6_global_km_per_cycle(df_full)

    # Sidebar: EV6 설정
    st.sidebar.header("EV6 설정")
    n_rows = len(df_full)
    idx_min, idx_max = 0, n_rows - 1

    current_idx = st.sidebar.slider(
        "EV6 로그 타임라인 인덱스",
        min_value=idx_min,
        max_value=idx_max,
        value=idx_max,
        step=1,
        help="뒤로 당기면 과거 시점 상태를 볼 수 있습니다. 기본은 가장 최근 샘플입니다.",
        key="ev6_index_slider",
    )

    avg_km_per_day = st.sidebar.number_input(
        "EV6 하루 평균 주행거리 가정 (km)",
        min_value=10.0,
        max_value=200.0,
        value=40.0,
        step=5.0,
        help="한국 승용차 평균은 대략 30~40 km/day 수준.",
        key="ev6_avg_km_per_day",
    )

    row = df_full.iloc[current_idx]
    ts = row.get("TimeStamp", None)

    soh = float(row.get("StateOfHealth", np.nan))
    soc = float(row.get("StateOfChargeBMS", np.nan))
    efc = float(row.get("EFC", np.nan))
    rul_cycles = float(row.get("RUL_cycles", np.nan))

    speed = float(row.get("Speed", np.nan))
    batt_temp = float(row.get("BatteryMaxTemperature", np.nan))
    out_temp = float(row.get("OutdoorTemperature", np.nan))
    temp_rise = float(row.get("temp_rise_ev6", np.nan))
    power_kw = float(row.get("power_kw", np.nan))

    # EFC 기준 3단계 RUL 시나리오 (선택 시점 기준)
    if np.isfinite(efc):
        rul_base_efc = max(0.0, DESIGN_EFC_EOL - efc)

        # 앞으로 조건이 가혹해지는 경우 (고온 + 고C-rate + DCFC 잦음 등)
        rul_worst = rul_base_efc * 0.7

        # 지금과 비슷한 운행/충전 패턴 유지
        rul_normal = rul_base_efc * 1.0

        # SOC 관리 + 고온/저온 회피 등으로 더 잘 관리하는 경우
        rul_best = rul_base_efc * 1.2
    else:
        rul_base_efc = np.nan
        rul_worst = rul_normal = rul_best = np.nan

    # SoH 기반 health
    if np.isfinite(soh):
        health_pct_ev6 = soh
    else:
        if np.isfinite(efc):
            used_frac = min(max(efc / DESIGN_EFC_EOL, 0.0), 1.0)
            health_pct_ev6 = 100.0 * (1.0 - used_frac)
        else:
            health_pct_ev6 = np.nan

    if np.isnan(health_pct_ev6):
        health_status_ev6, emoji_ev6, color_ev6 = "정보 부족", "⚪️", "#888888"
    elif health_pct_ev6 >= 90:
        health_status_ev6, emoji_ev6, color_ev6 = "매우 양호", "🟢", "#2ca02c"
    elif health_pct_ev6 >= 80:
        health_status_ev6, emoji_ev6, color_ev6 = "양호", "🟢", "#2ca02c"
    elif health_pct_ev6 >= 70:
        health_status_ev6, emoji_ev6, color_ev6 = "주의", "🟡", "#ff7f0e"
    else:
        health_status_ev6, emoji_ev6, color_ev6 = "교체/정비 고려", "🔴", "#d62728"

    if np.isfinite(rul_cycles) and np.isfinite(km_per_cycle_ev6):
        remaining_km_ev6 = rul_cycles * km_per_cycle_ev6
    else:
        remaining_km_ev6 = np.nan

    if np.isfinite(remaining_km_ev6) and avg_km_per_day > 0:
        remaining_days_ev6 = remaining_km_ev6 / avg_km_per_day
    else:
        remaining_days_ev6 = np.nan

    # Layout
    top_left_ev6, top_right_ev6 = st.columns([1.8, 2.2])
    bottom_left_ev6, bottom_right_ev6 = st.columns([2.2, 1.8])

    # TOP-LEFT: EV6 현재 상태
    with top_left_ev6:
        st.markdown(
            "<div style='font-size:20px; font-weight:700;'>EV6 – 현재(선택 시점) 상태</div>",
            unsafe_allow_html=True,
        )
        ts_str = ts.isoformat() if ts is not None and not pd.isna(ts) else "N/A"
        st.markdown(
            f"<div style='font-size:13px; color:#666;'>로그 시각 (UTC): {ts_str}</div>",
            unsafe_allow_html=True,
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            if np.isfinite(soh):
                st.metric("State of Health", f"{soh:.1f} %")
            else:
                st.metric("State of Health", "N/A")
        with c2:
            if np.isfinite(soc):
                st.metric("State of Charge (BMS)", f"{soc:.1f} %")
            else:
                st.metric("State of Charge (BMS)", "N/A")
        with c3:
            if np.isfinite(power_kw):
                st.metric("Instant Power", f"{power_kw:.1f} kW")
            else:
                st.metric("Instant Power", "N/A")

        st.markdown("---")

        c4, c5, c6 = st.columns(3)
        with c4:
            if np.isfinite(efc):
                st.metric("누적 EFC", f"{efc:.2f}")
            else:
                st.metric("누적 EFC", "N/A")
        with c5:
            if np.isfinite(rul_cycles):
                st.metric("예상 남은 수명", f"{rul_cycles:.1f} cycles")
            else:
                st.metric("예상 남은 수명", "N/A")
        with c6:
            if np.isfinite(health_pct_ev6):
                st.markdown(
                    f"<div style='font-size:16px; font-weight:600; color:{color_ev6};'>"
                    f"{emoji_ev6} 상태: {health_status_ev6} · {health_pct_ev6:.1f}%</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<div style='font-size:16px; font-weight:600; color:#888;'>"
                    "⚪️ 상태: 정보 부족</div>",
                    unsafe_allow_html=True,
                )

        # 조건별 RUL 시나리오 (EFC 기준)
        st.markdown("#### 조건별 RUL 시나리오 (EFC 기준)")

        s1, s2, s3 = st.columns(3)
        with s1:
            if np.isfinite(rul_worst):
                st.metric(
                    "가혹 사용 (worst)",
                    f"{rul_worst:.0f} cycles",
                    help="앞으로 고온/고C-rate/급가속·DCFC가 잦은 가혹 조건이 계속되는 경우",
                )
            else:
                st.metric("가혹 사용 (worst)", "N/A")

        with s2:
            if np.isfinite(rul_normal):
                st.metric(
                    "현재 수준 (normal)",
                    f"{rul_normal:.0f} cycles",
                    help="지금 1년치 로그와 비슷한 운행/충전 패턴을 유지하는 경우",
                )
            else:
                st.metric("현재 수준 (normal)", "N/A")

        with s3:
            if np.isfinite(rul_best):
                st.metric(
                    "관리 잘함 (best)",
                    f"{rul_best:.0f} cycles",
                    help="SOC 20~80% 관리, 고온/극저온 회피, DCFC 사용 줄이는 등 스트레스를 줄이는 경우",
                )
            else:
                st.metric("관리 잘함 (best)", "N/A")

        st.markdown(
            f"""
            <div style='font-size:12px; color:#000; margin-top:6px;'>
                - 전체 로그 기준 추정 1 cycle당 평균 주행거리: <b>{km_per_cycle_ev6:.1f} km/cycle</b><br/>
                - 현재 RUL 기준 예상 주행 가능 거리: <b>{remaining_km_ev6:,.0f} km</b><br/>
                - 하루 {avg_km_per_day:.0f} km 운행 가정 시, 
                  <b>약 {remaining_days_ev6:.1f}일</b> 동안 현재 수준의 배터리 성능을 기대할 수 있습니다.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        s1b, s2b, s3b = st.columns(3)
        with s1b:
            if np.isfinite(speed):
                st.metric("현재 속도", f"{speed:.1f} km/h")
            else:
                st.metric("현재 속도", "N/A")
        with s2b:
            if np.isfinite(out_temp):
                st.metric("외기 온도", f"{out_temp:.1f} °C")
            else:
                st.metric("외기 온도", "N/A")
        with s3b:
            if np.isfinite(batt_temp):
                txt = f"{batt_temp:.1f} °C"
                if np.isfinite(temp_rise):
                    txt += f" (ΔT ≈ {temp_rise:.1f}°C)"
                st.metric("배터리 최대 온도", txt)
            else:
                st.metric("배터리 최대 온도", "N/A")

    # TOP-RIGHT: EFC/RUL 시간 궤적
    with top_right_ev6:
        st.markdown("### EV6 – 시간에 따른 누적 EFC / RUL 변화")

        if "TimeStamp" not in df_full.columns or "EFC" not in df_full.columns:
            st.info("EFC 또는 TimeStamp 컬럼이 없어 궤적을 그릴 수 없습니다.")
        else:
            ts_all = df_full["TimeStamp"]
            efc_all = df_full["EFC"]
            rul_all = df_full.get("RUL_cycles", pd.Series([np.nan] * len(df_full)))

            fig = go.Figure()

            fig.add_trace(
                go.Scatter(
                    x=ts_all,
                    y=efc_all,
                    mode="lines",
                    name="누적 EFC",
                    line=dict(width=2),
                    yaxis="y1",
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=ts_all,
                    y=rul_all,
                    mode="lines",
                    name="예상 RUL (cycles)",
                    line=dict(dash="dash"),
                    yaxis="y2",
                )
            )

            if ts is not None and not pd.isna(ts):
                fig.add_vline(
                    x=ts,
                    line_dash="dot",
                    line_width=2,
                    line_color="black",
                    annotation_text="현재 선택 시점",
                    annotation_position="top",
                )

            fig.update_layout(
                margin=dict(l=50, r=50, t=40, b=40),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
                height=380,
                xaxis=dict(title="Time"),
                yaxis=dict(
                    title="누적 EFC",
                    side="left",
                    showgrid=True,
                ),
                yaxis2=dict(
                    title="RUL (cycles)",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                ),
            )
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "좌측 축: 누적 EFC (충·방전 등가 풀사이클 수), "
                "우측 축: 간단한 EFC/SoH 기반 잔여 수명 근사 RUL (cycles)."
            )

    # BOTTOM-LEFT: EV6 일별 요약
    with bottom_left_ev6:
        st.markdown("### EV6 – 일별 주행 및 효율 요약")

        if df_daily.empty:
            st.info("ev6_daily_summary.csv 내용이 비어 있습니다.")
        else:
            all_dates = df_daily["date"].tolist()
            sel_date = st.selectbox(
                "날짜 선택",
                options=all_dates,
                index=len(all_dates) - 1,
            )

            row_d = df_daily[df_daily["date"] == sel_date].iloc[0]

            d1, d2, d3, d4 = st.columns(4)
            with d1:
                st.metric("일일 주행거리", f"{row_d.get('distance_km', np.nan):.2f} km")
            with d2:
                st.metric(
                    "평균 속도",
                    f"{row_d.get('Speed_mean', np.nan):.1f} km/h",
                )
            with d3:
                soc_mean = row_d.get("StateOfChargeBMS_mean", np.nan)
                if np.isfinite(soc_mean):
                    st.metric("평균 SoC (BMS)", f"{soc_mean:.1f} %")
                else:
                    st.metric("평균 SoC (BMS)", "N/A")
            with d4:
                eff = row_d.get("kWh_100km_mean", np.nan)
                if np.isfinite(eff):
                    st.metric("평균 효율", f"{eff:.1f} kWh/100km")
                else:
                    st.metric("평균 효율", "N/A")

            st.markdown("---")

            fig2 = go.Figure()
            fig2.add_trace(
                go.Scatter(
                    x=df_daily["date"],
                    y=df_daily["distance_km"],
                    mode="lines+markers",
                    name="일일 주행거리 (km)",
                )
            )

            if "kWh_100km_mean" in df_daily.columns:
                fig2.add_trace(
                    go.Scatter(
                        x=df_daily["date"],
                        y=df_daily["kWh_100km_mean"],
                        mode="lines+markers",
                        name="평균 효율 (kWh/100km)",
                        yaxis="y2",
                    )
                )

            fig2.update_layout(
                margin=dict(l=50, r=50, t=20, b=40),
                height=320,
                xaxis=dict(title="날짜"),
                yaxis=dict(title="일일 주행거리 (km)", side="left", showgrid=True),
                yaxis2=dict(
                    title="평균 효율 (kWh/100km)",
                    overlaying="y",
                    side="right",
                    showgrid=False,
                ),
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
            )
            st.plotly_chart(fig2, use_container_width=True)

    # BOTTOM-RIGHT: EV6 설명
    with bottom_right_ev6:
        st.markdown("### 해석 메모 (EV6)")

        st.markdown(
            f"""
            - **EFC (Equivalent Full Cycles)**  
              EV6 누적 방전 에너지(CED)를 기준으로, 사용 가능한 용량을 약 {USABLE_KWH:.1f} kWh로 보고  
              몇 번의 풀 충·방전 사이클과 동등한지 나타낸 값입니다.

            - **RUL (예상 남은 수명, cycles)**  
              설계상 약 {DESIGN_EFC_EOL:.0f} EFC 부근에서 SoH가 {SOH_EOL:.0f}%에 도달한다고 가정하고,  
              현재까지의 EFC와 SoH(StateOfHealth)에서 남은 사이클 수를 단순 근사합니다.  
              OEM 내부 보증 모델이 아니라, **실제 EV6 주행 로그에 기반한 거친 지표**로 보는 것이 맞습니다.

            - **조건별 RUL 시나리오 (worst/normal/best)**  
              같은 현재 EFC 기준에서, 앞으로 운행/충전이 얼마나 가혹해지는지에 따라  
              대략 0.7배 (가혹), 1.0배 (현재 수준), 1.2배 (관리 잘함) 정도의 범위를  
              단순 가정하여 3단계로 나눈 잔여 수명 시나리오입니다.

            - **km per cycle 근사**  
              전체 EV6 로그에서의 총 주행거리 ≈ {total_km:.1f} km,  
              총 EFC로 나누어 ≈ {km_per_cycle_ev6:.1f} km/사이클 정도로 추정했습니다.  
              이 값을 NASA 탭에서 scale 감각용으로 같이 사용하는 것도 가능합니다.

            - 이 EV6 탭은 NASA 셀 기반 BMAML 모델과 별개로,  
              "**실제 도로 주행에서의 배터리 사용 패턴이 어느 정도 cycle 스케일인지**"를  
              숫자로 체감하기 위한 보조 뷰로 쓰면 좋습니다.
            """
        )
