import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_TITLE = "Fleet Battery Health APM — RUL Dashboard (Overview only)"
DEFAULT_DB_FILENAME = "battery_training_data_cleaned_final_causal.csv"
ENV_DB_PATH_KEY = "BATTERY_FINAL_DB_CSV"


def resolve_db_path(user_path: str) -> Path:
    """
    Resolve DB CSV path robustly.

    Priority:
      1) explicit user_path (sidebar)
      2) env var BATTERY_FINAL_DB_CSV
      3) common project-relative locations (based on this script file)
      4) cwd-relative locations
    """
    tried = []
    # 1) explicit
    if user_path:
        p = Path(user_path).expanduser()
        tried.append(p)
        if p.exists():
            return p

    # 2) env
    env_val = (os.environ.get(ENV_DB_PATH_KEY) or "").strip()
    if env_val:
        p = Path(env_val).expanduser()
        tried.append(p)
        if p.exists():
            return p

    # 3) script-relative
    here = Path(__file__).resolve()
    candidates = [
        here.parent / DEFAULT_DB_FILENAME,                       # same folder
        here.parent.parent / DEFAULT_DB_FILENAME,                # parent folder
        here.parent.parent / "db" / DEFAULT_DB_FILENAME,         # deep_learning/db
        here.parent.parent.parent / "db" / DEFAULT_DB_FILENAME,  # backend/deep_learning/db (if core nested)
        here.parent.parent / "data" / DEFAULT_DB_FILENAME,
    ]

    # 4) cwd-relative
    cwd = Path.cwd().resolve()
    candidates += [
        cwd / DEFAULT_DB_FILENAME,
        cwd / "db" / DEFAULT_DB_FILENAME,
        cwd / "data" / DEFAULT_DB_FILENAME,
        cwd.parent / "db" / DEFAULT_DB_FILENAME,
    ]

    for c in candidates:
        tried.append(c)
        if c.exists():
            return c

    # Nothing found -> raise with tried paths
    raise FileNotFoundError("DB CSV not found. Tried:\n" + "\n".join(str(t) for t in tried))



def _badge(label: str, color: str) -> str:
    return f"<span style='padding:2px 10px;border-radius:999px;background:{color};color:white;font-weight:700;font-size:0.85rem'>{label}</span>"


def load_db(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # normalize key columns
    if "battery" in df.columns and "battery_id" not in df.columns:
        df = df.rename(columns={"battery": "battery_id"})
    if "cycle_num" not in df.columns and "cycle" in df.columns:
        df = df.rename(columns={"cycle": "cycle_num"})
    return df


def get_risk_label(initial_rul: float, current_rul: float):
    if not (np.isfinite(initial_rul) and initial_rul > 0 and np.isfinite(current_rul)):
        return "UNKNOWN", "#6b7280", float("nan")
    rul_pct = 100.0 * float(current_rul) / float(initial_rul)
    if rul_pct >= 60.0:
        return "OK", "#16a34a", rul_pct
    if rul_pct >= 30.0:
        return "WARN", "#f59e0b", rul_pct
    return "ALERT", "#dc2626", rul_pct


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Quick verification build: battery selection + RUL trajectory + key metrics. (Other tabs intentionally disabled)")

    with st.sidebar:
        st.header("Data")
        db_hint = os.environ.get(ENV_DB_PATH_KEY, "").strip()
        st.caption(f"Tip: set env var `{ENV_DB_PATH_KEY}` to avoid typing the path every time.")
        db_path_text = st.text_input("Final DB CSV path", value=db_hint)
        try:
            db_path = resolve_db_path(db_path_text)
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()

        st.success(f"Loaded: {db_path}")
        df = load_db(db_path)

        bids = sorted(df["battery_id"].astype(str).unique().tolist())
        selected_bid = st.selectbox("Battery", bids, index=0)

        sdf = df[df["battery_id"].astype(str) == str(selected_bid)].copy()
        sdf = sdf.sort_values("cycle_num")
        cycles = sdf["cycle_num"].to_numpy(dtype=int)

        min_c = int(cycles.min()) if len(cycles) else 0
        max_c = int(cycles.max()) if len(cycles) else 0
        current_cycle = st.slider("Cycle cursor", min_value=min_c, max_value=max_c, value=min_c, step=1)

    if sdf.empty:
        st.warning("No records for the selected battery.")
        st.stop()

    # nearest record to cursor (cycle_num should exist but keep robust)
    idx = int(np.argmin(np.abs(cycles - current_cycle)))
    row = sdf.iloc[idx]

    # RUL series (ground-truth in this DB)
    rul = sdf["rul_cycles"].to_numpy(dtype=float) if "rul_cycles" in sdf.columns else np.full(len(sdf), np.nan)
    initial_rul = float(rul[0]) if len(rul) else float("nan")
    current_rul = float(rul[idx]) if len(rul) else float("nan")
    risk_label, risk_color, rul_pct = get_risk_label(initial_rul, current_rul)

    # Header
    st.markdown(f"## Battery {selected_bid}  &nbsp; {_badge(risk_label, risk_color)}", unsafe_allow_html=True)

    if str(selected_bid) == "B0043":
        st.caption(
            "After EOL / severe degradation, feature distribution shifts (e.g., temperature/self-heating) can push the model out of its training manifold, "
            "causing non-physical RUL rebounds. Treat predictions in this region as unreliable."
        )

    # Layout
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.subheader("RUL trajectory (cycles remaining)")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cycles, y=rul, mode="lines", name="RUL (ground truth)"))
        fig.add_vline(x=current_cycle, line_dash="dash", line_width=2)
        fig.update_layout(
            height=420,
            margin=dict(l=40, r=20, t=20, b=40),
            xaxis_title="Cycle",
            yaxis_title="RUL (cycles)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Key metrics at cursor")

        def metric(label, value, fmt="{:.4f}"):
            if value is None or (isinstance(value, float) and not np.isfinite(value)):
                st.write(f"**{label}:** —")
                return
            if isinstance(value, (int, np.integer)):
                st.write(f"**{label}:** {int(value)}")
            elif isinstance(value, (float, np.floating)):
                st.write(f"**{label}:** {fmt.format(float(value))}")
            else:
                st.write(f"**{label}:** {value}")

        metric("Cycle cursor", int(current_cycle), "{}")
        if np.isfinite(rul_pct):
            metric("RUL% vs initial", rul_pct, "{:.1f}%")
        metric("RUL (cycles remaining)", current_rul, "{:.1f}")
        metric("SoH", row.get("soh", np.nan), "{:.3f}")
        metric("Capacity mean (Ah)", row.get("capacity_mean", np.nan), "{:.4f}")
        metric("Re (stored as dcr)", row.get("dcr", np.nan), "{:.4f}")
        metric("Impedance sum (Re + Rct)", row.get("impedance_sum", np.nan), "{:.4f}")

        # The confusing one: ambient_temp_c in your pipeline is actually Temperature_measured[0]
        metric("Temperature_measured[0], cell temp start (°C)", row.get("ambient_temp_c", np.nan), "{:.2f}")
        metric("Temperature mean (°C)", row.get("temperature_mean", np.nan), "{:.2f}")
        metric("Thermal stress", row.get("thermal_stress", np.nan), "{:.3f}")
        metric("LLI", row.get("lli", np.nan), "{:.4f}")
        metric("LAM", row.get("lam", np.nan), "{:.4f}")

        st.divider()
        st.caption(
            "Note: this verification build reads **battery_training_data_cleaned_final_causal.csv** and plots the ground-truth RUL field `rul_cycles`.\n"
            "Full APM tabs (Monitoring / Compare / What-if) will be re-enabled after this page is confirmed working."
        )


if __name__ == "__main__":
    main()
