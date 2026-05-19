import math
import sys
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ev6_senario_rul import (
    compute_ev6_scenario_features,
    estimate_rul_from_scenario,
    load_feature_stats,
    DESIGN_EFC_EOL,
)

# ----------------------------------------
# 경로 & 패키지 설정 (app_rul_dashboard.py와 동일 패턴)
# ----------------------------------------
FILE_DIR = Path(__file__).resolve().parent      # .../core/pages
CORE_DIR = FILE_DIR.parent                      # .../core
PROJECT_ROOT = CORE_DIR.parent                  # .../v11

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# EV6 전처리 유틸 (이미 v11/deep_learning/core/real_time.py에 있음)

# Ensure backend root is on sys.path so 'deep_learning' package is importable
_THIS_FILE = Path(__file__).resolve()
_BACKEND_ROOT = _THIS_FILE.parent.parent  # .../backend
if str(_BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(_BACKEND_ROOT))

from deep_learning.core import real_time as ev6rt  # type: ignore

EV6_DIR = CORE_DIR / "real_time"
EV6_CSV = EV6_DIR / "decoded_ev6_data_full.csv"
FEATURE_STATS_PATH = CORE_DIR / "analysis" / "feature_rul_stats.json"

# ----------------------------------------
# Streamlit 설정
# ----------------------------------------
st.set_page_config(
    page_title="EV6 Real Driving RUL Dashboard",
    layout="wide",
)

st.title("🚗 Kia EV6 Real Driving · RUL & Efficiency")
st.caption("CSS EV6 CAN 로그 + 간단 EFC/SoH RUL 근사 모델 기반 대시보드")

# ----------------------------------------
# 헬퍼: 시간 축 downsample
# ----------------------------------------
def make_downsampled_ts(
    df: pd.DataFrame,
    time_col: str,
    value_cols: list[str],
    rule: str = "10min",
    max_points: int = 3000,
) -> pd.DataFrame | None:
    """
    - time_col 기준으로 DateTimeIndex 만들고
    - rule 단위로 resample(mean)
    - max_points 넘으면 균등 샘플링으로 줄이기
    """
    if time_col not in df.columns:
        return None

    use_cols = [c for c in value_cols if c in df.columns]
    if not use_cols:
        return None

    tmp = df[[time_col] + use_cols].dropna().copy()
    if tmp.empty:
        return None

    # DateTimeIndex 세팅
    tmp = tmp.set_index(time_col)
    if not isinstance(tmp.index, pd.DatetimeIndex):
        try:
            tmp.index = pd.to_datetime(tmp.index, errors="coerce")
            tmp = tmp[tmp.index.notna()]
        except Exception:
            return None

    if tmp.empty:
        return None

    # 시간 기준 resample
    if rule is not None:
        tmp = tmp.resample(rule).mean()

    # 포인트 수 너무 많으면 균등 샘플링
    n = len(tmp)
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points).astype(int)
        tmp = tmp.iloc[idx]

    return tmp


# ----------------------------------------
# EV6 데이터 로드 & 전처리 (캐시)
# ----------------------------------------
@st.cache_data(show_spinner="EV6 로그 로딩 및 전처리 중...")
def load_ev6_processed(csv_path: Path):
    """
    real_time.py 유틸 재사용:
      - CSV 로드
      - 효율(kWh/100km) 계산
      - EFC & RUL 계산
      - 파생 물리 피처 추가 (power, C-rate, DoD, temp_rise)
      - 일(day) 단위 summary 생성
    """
    if not csv_path.exists():
        return None, None

    df = ev6rt.load_ev6_csv(csv_path)
    df = ev6rt.add_efficiency_column(df)
    df = ev6rt.add_efc_and_rul(df)
    df = ev6rt.add_ev6_derived_physics(df, pack_ah=120.6)  # 롱레인지 기준 Ah
    daily = ev6rt.build_daily_summary(df)
    return df, daily


df_full, df_daily = load_ev6_processed(EV6_CSV)

if df_full is None or df_daily is None:
    st.error(
        "EV6 CSV를 찾을 수 없습니다.\n\n"
        f"경로를 확인해 주세요:\n{EV6_CSV}"
    )
    st.stop()

# ----------------------------------------
# 사이드바: 기간 선택 (빠른 프리셋 + 사용자 지정)
# ----------------------------------------
min_date = df_daily["date"].min()
max_date = df_daily["date"].max()

default_start = max(min_date, max_date - dt.timedelta(days=30))

with st.sidebar:
    st.header("EV6 데이터 필터")

    preset = st.radio(
        "빠른 기간 선택",
        ("전체 기간", "최근 7일", "최근 30일", "최근 90일", "사용자 지정"),
        index=2,  # 기본: 최근 30일
    )

    if preset == "사용자 지정":
        # 사용자가 직접 시작/종료일 지정
        date_range = st.date_input(
            "기간 선택",
            value=(default_start, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        if isinstance(date_range, tuple):
            start_date, end_date = date_range
        else:
            start_date = default_start
            end_date = date_range
    else:
        # 프리셋 기준 자동 설정
        end_date = max_date
        if preset == "전체 기간":
            start_date = min_date
        elif preset == "최근 7일":
            start_date = max(min_date, max_date - dt.timedelta(days=7))
        elif preset == "최근 30일":
            start_date = max(min_date, max_date - dt.timedelta(days=30))
        elif preset == "최근 90일":
            start_date = max(min_date, max_date - dt.timedelta(days=90))

# 일 단위 요약에서 기간 필터
mask_daily = (df_daily["date"] >= start_date) & (df_daily["date"] <= end_date)
daily_sel = df_daily.loc[mask_daily].copy()

if daily_sel.empty:
    st.warning("선택한 기간에 대한 일 단위 데이터가 없습니다.")
    st.stop()

# raw 로그에서도 같은 기간으로 필터
if "TimeStamp" in df_full.columns:
    df_full["date_only"] = df_full["TimeStamp"].dt.date
    mask_full = (df_full["date_only"] >= start_date) & (df_full["date_only"] <= end_date)
    df_full_sel = df_full.loc[mask_full].copy()
else:
    df_full_sel = df_full.copy()

# ----------------------------------------
# 글로벌 요약 지표
# ----------------------------------------

# EFC 범위 (선택 기간 기준)
if "EFC" in df_full_sel.columns:
    efc_series = df_full_sel["EFC"].dropna()
else:
    efc_series = pd.Series(dtype=float)

efc_min = float(efc_series.min()) if not efc_series.empty else 0.0
efc_max = float(efc_series.max()) if not efc_series.empty else 0.0
efc_span = max(0.0, efc_max - efc_min)

# 마지막 사이클 기준 RUL (선택 기간 마지막 샘플)
if "RUL_cycles" in df_full_sel.columns:
    rul_series_sel = df_full_sel["RUL_cycles"].dropna()
else:
    rul_series_sel = pd.Series(dtype=float)

last_cycle_rul = float(rul_series_sel.iloc[-1]) if not rul_series_sel.empty else float("nan")

# 전체 로그 기준: 마지막 TimeStamp / EFC
if "TimeStamp" in df_full.columns:
    last_ts = df_full["TimeStamp"].max()
    last_date = last_ts.date()
else:
    last_ts = None
    last_date = None

if "EFC" in df_full.columns:
    efc_last = float(df_full["EFC"].dropna().max())
else:
    efc_last = float("nan")

# 오늘 날짜 (현재 시점)
today = dt.date.today()
today_str = today.strftime("%Y/%m/%d")

# 마지막 로그 이후 경과 일수
if last_date is not None:
    days_gap = max(0, (today - last_date).days)
else:
    days_gap = 0

# 최근 90일 기준 하루 평균 EFC 증가량 추정
# (로그가 90일보다 짧으면 전체 구간 사용)
efc_per_day = 0.0
if "TimeStamp" in df_full.columns and "EFC" in df_full.columns:
    df_full_sorted = df_full.sort_values("TimeStamp")
    if last_ts is not None:
        start_recent = last_ts - pd.Timedelta(days=90)
        df_recent = df_full_sorted[df_full_sorted["TimeStamp"] >= start_recent]
        if df_recent.empty:
            df_recent = df_full_sorted

        efc_vals = df_recent["EFC"].astype(float).to_numpy()
        t_vals = df_recent["TimeStamp"].to_numpy()
        mask_valid = np.isfinite(efc_vals)
        efc_vals = efc_vals[mask_valid]
        t_vals = t_vals[mask_valid]

        if efc_vals.size > 1:
            efc_delta = float(efc_vals[-1] - efc_vals[0])
            days_delta = max(
                1.0,
                (t_vals[-1] - t_vals[0]) / np.timedelta64(1, "D"),
            )
            efc_per_day = max(0.0, efc_delta / days_delta)

# 👉 오늘 시점까지의 EFC 추정
#    = 마지막 로그 EFC + (최근 평균 EFC/day * (오늘 - 마지막 로그 날짜))
if math.isfinite(efc_last):
    efc_today = efc_last + efc_per_day * days_gap
    efc_today = float(np.clip(efc_today, 0.0, DESIGN_EFC_EOL))
else:
    efc_today = float("nan")

# NASA 시나리오 모델 기반 현재 시점 RUL 추정 (3단계 분해)
rul_efc_today = float("nan")
rul_scenario_raw_today = float("nan")
rul_scenario_today = float("nan")

try:
    feature_stats = load_feature_stats(FEATURE_STATS_PATH)
    # 사용 패턴은 "선택 기간" df_full_sel 기준으로 본다
    ev6_feats = compute_ev6_scenario_features(df_full_sel)

    if math.isfinite(efc_today):
        res = estimate_rul_from_scenario(
            ev6_feats=ev6_feats,
            feature_stats=feature_stats,
            baseline_feats=None,
            baseline_rul=DESIGN_EFC_EOL,
            efc_current=efc_today,
            factor_min=0.5,
            factor_max=1.0,  # 🔴 여기서 보너스 안 줌: EFC 기반 RUL보다 커지지 않게
        )
        rul_efc_today = res.get("rul_from_efc", float("nan"))
        rul_scenario_raw_today = res.get("rul_scenario_raw", float("nan"))
        rul_scenario_today = res.get("rul_scenario", float("nan"))
except Exception:
    # NASA 모델 계산 실패해도 대시보드 전체 죽지 않게 방어
    pass

# 선택 기간 총 주행 거리 / 효율
total_dist = float(daily_sel["distance_km"].sum())
if "kWh_100km_mean" in daily_sel.columns:
    mean_eff = float(daily_sel["kWh_100km_mean"].dropna().mean())
else:
    mean_eff = float("nan")

# 상단 메트릭: 기간 요약 + "마지막 사이클" RUL
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("선택 기간 총 주행거리 (km)", f"{total_dist:,.1f}")
with c2:
    st.metric(
        "평균 효율 (kWh/100km)",
        f"{mean_eff:,.1f}" if math.isfinite(mean_eff) else "N/A",
    )
with c3:
    st.metric("누적 EFC (선택 기간)", f"{efc_span:.2f}")
with c4:
    st.metric(
        "마지막 사이클 추정 RUL (cycles)",
        f"{last_cycle_rul:.1f}" if math.isfinite(last_cycle_rul) else "N/A",
    )

# 🔻 여기부터 네가 원한 "현재 시점(오늘 날짜) 추정 RUL – 3단계" 세로 표시
st.markdown(f"### 현재 시점 ({today_str}) 추정 RUL (내 NASA 시나리오 모델 기준)")

st.metric(
    "현재 시점 추정 RUL (cycles)",
    f"{rul_scenario_today:.1f}" if math.isfinite(rul_scenario_today) else "N/A",
)


# ----------------------------------------
# 레이아웃: 2 x 2
# ----------------------------------------
top_left, top_right = st.columns(2)
bottom_left, bottom_right = st.columns(2)

# 1) 일별 주행거리 & 효율
with top_left:
    st.markdown("### 일별 주행거리 & 효율")

    plot_cols = ["distance_km"]
    if "kWh_100km_mean" in daily_sel.columns:
        plot_cols.append("kWh_100km_mean")

    st.line_chart(
        daily_sel.set_index("date")[plot_cols],
        height=260,
    )

# 2) SoC / SoH (일 평균)
with top_right:
    st.markdown("### SoC / SoH (일 평균)")

    cols = []
    if "StateOfChargeBMS_mean" in daily_sel.columns:
        cols.append("StateOfChargeBMS_mean")
    if "StateOfHealth_mean" in daily_sel.columns:
        cols.append("StateOfHealth_mean")

    if cols:
        st.line_chart(
            daily_sel.set_index("date")[cols],
            height=260,
        )
    else:
        st.info(
            "일 평균 SoC/SoH 컬럼이 없습니다. "
            "real_time.py에서 daily summary에 포함하도록 수정할 수 있습니다."
        )

# 3) 시간 축 기반 EFC / RUL 궤적 (downsample 포함, Plotly 2축)
with bottom_left:
    st.markdown("### EV6 EFC · RUL 궤적 (선택 기간, 10분 downsample)")

    ts = make_downsampled_ts(
        df_full_sel,
        time_col="TimeStamp",
        value_cols=["EFC", "RUL_cycles"],
        rule="10min",      # 10분 단위 resample
        max_points=3000,   # 너무 많으면 균등 간격 샘플
    )

    if ts is not None and not ts.empty:
        ts = ts.sort_index()

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # EFC (왼쪽 y축)
        if "EFC" in ts.columns:
            y1 = ts["EFC"].to_numpy(dtype=float)
            mask1 = np.isfinite(y1)
            fig.add_trace(
                go.Scatter(
                    x=ts.index[mask1],
                    y=y1[mask1],
                    mode="lines",
                    name="EFC",
                ),
                secondary_y=False,
            )

            if mask1.any():
                y1_valid = y1[mask1]
                m1 = float(y1_valid.min())
                M1 = float(y1_valid.max())
                margin1 = 0.05 * max(1.0, M1 - m1)
                fig.update_yaxes(
                    range=[m1 - margin1, M1 + margin1],
                    title_text="EFC",
                    secondary_y=False,
                )

        # RUL (오른쪽 y축)
        if "RUL_cycles" in ts.columns:
            y2 = ts["RUL_cycles"].to_numpy(dtype=float)
            mask2 = np.isfinite(y2)
            fig.add_trace(
                go.Scatter(
                    x=ts.index[mask2],
                    y=y2[mask2],
                    mode="lines",
                    name="RUL_cycles",
                ),
                secondary_y=True,
            )

            if mask2.any():
                y2_valid = y2[mask2]
                m2 = float(y2_valid.min())
                M2 = float(y2_valid.max())
                margin2 = 0.05 * max(1.0, M2 - m2)
                fig.update_yaxes(
                    range=[m2 - margin2, M2 + margin2],
                    title_text="RUL (cycles)",
                    secondary_y=True,
                )

        fig.update_layout(
            margin=dict(l=40, r=40, t=40, b=40),
            height=260,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
            ),
            xaxis_title="Time",
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("선택 기간에 대한 EFC/RUL 궤적을 그릴 수 없습니다.")

# 4) 파생 물리 피처 (C-rate, 온도 상승, 전력) 시간 축 플롯 (downsample)
with bottom_right:
    st.markdown("### 파생 물리 피처 (C-rate, 온도 상승, 전력)")

    phys_cols = [c for c in ["eff_c_rate", "temp_rise_ev6", "power_kw"] if c in df_full_sel.columns]

    if phys_cols:
        ts_phys = make_downsampled_ts(
            df_full_sel,
            time_col="TimeStamp",
            value_cols=phys_cols,
            rule="10min",
            max_points=3000,
        )
        if ts_phys is not None and not ts_phys.empty:
            st.line_chart(ts_phys, height=260)
        else:
            st.info("선택 기간에 대해 파생 피처를 그릴 수 없습니다.")
    else:
        st.info(
            "eff_c_rate / temp_rise_ev6 / power_kw 컬럼이 없습니다. "
            "real_time.py에서 파생 피처 추가 여부를 확인하세요."
        )

# ----------------------------------------
# 설명 블록
# ----------------------------------------
st.markdown("---")
st.markdown(
    """
    - **EFC (Equivalent Full Cycles)**  
      누적 방전 에너지를 EV6 유효 용량(약 74.6 kWh 기준)으로 나눈 값으로,  
      대략 몇 번의 완전 충·방전 사이클을 돌렸는지 보여줍니다.

    - **RUL_cycles**  
      EFC와 SoH(상태 건강도)를 동시에 고려한 간단한 잔여 수명 근사치입니다.  
      (설계상 1000 EFC에서 SoH 80% 도달한다고 가정한 물리 직관 기반 모델)

    - 이 페이지는 CSS EV6 실주행 로그를 기반으로,  
      NASA BMAML RUL 대시보드와 어울리는 **실차 로그 해석용 보조 뷰** 역할을 합니다.
    """
)
 