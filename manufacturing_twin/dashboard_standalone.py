"""
dashboard.py

Streamlit UI for the manufacturing digital twin. Replays the sensor
stream, updates the twin, and shows:
  - current machine health + failure probability
  - current operating conditions
  - a running chart of failure probability over time
  - the exact JSON evidence packet this twin would hand off to the DSS

Run with:  streamlit run dashboard.py
"""

import sys
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import stream
import twin as twin_module

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import ui_theme
from ui_theme import ACCENT, NAVY, WARN, BAD, GOOD, LINE, MUTED, show_plotly

st.set_page_config(page_title="Manufacturing Digital Twin", layout="wide")
ui_theme.inject()

DEFAULT_MACHINE_ID = "CNC-01"

HEALTH_COLORS = {
    "Healthy": GOOD,
    "Warning": WARN,
    "Critical": BAD,
}

# --- Session state: persists the twin + stream position across reruns --
if "twin" not in st.session_state:
    st.session_state.twin = twin_module.MachineTwin(DEFAULT_MACHINE_ID)
if "rows" not in st.session_state:
    st.session_state.rows = stream.load_stream().to_dict(orient="records")
if "index" not in st.session_state:
    st.session_state.index = 0
if "playing" not in st.session_state:
    st.session_state.playing = False

total_rows = len(st.session_state.rows)

# --- Sidebar controls ----------------------------------------------------
st.sidebar.markdown('<div class="eyebrow">Decision Support Console</div>', unsafe_allow_html=True)
st.sidebar.title("Manufacturing Twin")
st.sidebar.caption(
    f"Machine **{DEFAULT_MACHINE_ID}** — replayed sensor stream with XGBoost "
    "failure prediction. Evidence source for the DSS."
)

speed = st.sidebar.slider("Playback delay (seconds/reading)", 0.0, 1.0, 0.1, 0.05)
step_size = st.sidebar.slider("Readings per step", 1, 50, 1)

col_play, col_reset = st.sidebar.columns(2)
if col_play.button("Play" if not st.session_state.playing else "Pause"):
    st.session_state.playing = not st.session_state.playing
if col_reset.button("Reset"):
    st.session_state.twin = twin_module.MachineTwin(DEFAULT_MACHINE_ID)
    st.session_state.index = 0
    st.session_state.playing = False

progress = min(st.session_state.index / max(total_rows, 1), 1.0)
st.sidebar.progress(progress)
st.sidebar.caption(f"Reading {st.session_state.index} / {total_rows}")

# --- Advance the stream ----------------------------------------------------
if st.session_state.index < total_rows:
    for _ in range(step_size):
        if st.session_state.index >= total_rows:
            break
        reading = st.session_state.rows[st.session_state.index]
        st.session_state.twin.update(reading)
        st.session_state.index += 1

state = st.session_state.twin.latest_state

# --- Header ----------------------------------------------------------------
st.markdown(
    '<div class="eyebrow">FYP · AI-Based DSS for Digital Twin-Driven Transformation</div>',
    unsafe_allow_html=True,
)
st.title("Manufacturing Digital Twin")
st.caption(
    "Lightweight prototype: a live mirror of one machine that scores each "
    "sensor reading and hands the DSS a compact evidence packet — not raw telemetry."
)

if state is None:
    st.info("No readings processed yet. Click Play or Reset to start the stream.")
    st.stop()

# --- Top-line status ---------------------------------------------------------
health = state["health"]
color = HEALTH_COLORS.get(health, MUTED)

k1, k2, k3, k4 = st.columns(4)
with k1:
    st.markdown(
        f"""
        <div class="health-banner" style="background:{color}14;border-color:{color};">
            <div class="k">Machine health</div>
            <div class="v" style="color:{color};">{health}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with k2:
    st.metric("Failure probability", f"{state['failure_probability']*100:.2f}%")
with k3:
    st.metric("Machine", state["machine"], help=f"Type: {state['machine_type']}")
with k4:
    st.metric("Stream progress", f"{st.session_state.index} / {total_rows}")

if health != "Healthy":
    st.warning(f"**Recommendation:** {state['recommendation']}")
else:
    st.success(f"**Recommendation:** {state['recommendation']}")

# --- Operating conditions ---------------------------------------------------
st.subheader("Current operating conditions")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Air temperature", f"{state['air_temperature_K']} K")
c2.metric("Process temperature", f"{state['process_temperature_K']} K")
c3.metric("Rotational speed", f"{state['rotational_speed_rpm']} rpm")
c4.metric("Torque", f"{state['torque_Nm']} Nm")
c5.metric("Tool wear", f"{state['tool_wear_min']} min")

hist_df = pd.DataFrame(st.session_state.twin.history)

chart_col, table_col = st.columns([1.6, 1], gap="large")

with chart_col:
    st.subheader("Failure probability over time")
    if not hist_df.empty:
        y = hist_df["failure_probability"] * 100
        line_color = color
        fig = go.Figure()
        fig.add_hrect(y0=60, y1=100, fillcolor=BAD, opacity=0.08, line_width=0)
        fig.add_hrect(y0=30, y1=60, fillcolor=WARN, opacity=0.08, line_width=0)
        fig.add_hrect(y0=0, y1=30, fillcolor=GOOD, opacity=0.08, line_width=0)
        fig.add_hline(y=30, line_dash="dot", line_color=GOOD,
                      annotation_text="Healthy < 30%", annotation_font_color=GOOD)
        fig.add_hline(y=60, line_dash="dot", line_color=WARN,
                      annotation_text="Critical ≥ 60%", annotation_font_color=WARN)
        fig.add_trace(go.Scatter(
            y=y,
            x=list(range(1, len(y) + 1)),
            mode="lines",
            name="Failure probability",
            line=dict(color=line_color, width=2),
            hovertemplate="Reading %{x}<br>%{y:.2f}%<extra></extra>",
        ))
        fig.update_layout(
            showlegend=False,
            yaxis_title="Failure probability (%)",
            xaxis_title="Reading",
            yaxis_range=[0, max(100, float(y.max()) + 5)],
        )
        show_plotly(fig, height=360)
    else:
        st.info("History will appear as the stream advances.")

    if not hist_df.empty and len(hist_df) > 1:
        st.subheader("Sensor traces")
        sensors = make_subplots(
            rows=2, cols=2,
            subplot_titles=("Temperatures (K)", "Rotational speed (rpm)",
                            "Torque (Nm)", "Tool wear (min)"),
            vertical_spacing=0.16, horizontal_spacing=0.1,
        )
        x = list(range(1, len(hist_df) + 1))
        sensors.add_trace(go.Scatter(x=x, y=hist_df["air_temperature_K"],
                                     name="Air", line=dict(color=NAVY, width=1.4)),
                          row=1, col=1)
        sensors.add_trace(go.Scatter(x=x, y=hist_df["process_temperature_K"],
                                     name="Process", line=dict(color=ACCENT, width=1.4)),
                          row=1, col=1)
        sensors.add_trace(go.Scatter(x=x, y=hist_df["rotational_speed_rpm"],
                                     name="Speed", line=dict(color=ACCENT, width=1.4),
                                     showlegend=False),
                          row=1, col=2)
        sensors.add_trace(go.Scatter(x=x, y=hist_df["torque_Nm"],
                                     name="Torque", line=dict(color=WARN, width=1.4),
                                     showlegend=False),
                          row=2, col=1)
        sensors.add_trace(go.Scatter(x=x, y=hist_df["tool_wear_min"],
                                     name="Wear", line=dict(color=BAD, width=1.4),
                                     showlegend=False),
                          row=2, col=2)
        sensors.update_layout(legend=dict(orientation="h", y=1.12))
        show_plotly(sensors, height=420)

with table_col:
    st.subheader("Recent readings")
    if not hist_df.empty:
        recent = hist_df.tail(12)[
            ["health", "failure_probability", "air_temperature_K",
             "process_temperature_K", "rotational_speed_rpm", "torque_Nm", "tool_wear_min"]
        ].iloc[::-1].copy()
        recent["failure_probability"] = (recent["failure_probability"] * 100).round(2)
        recent = recent.rename(columns={
            "health": "Health",
            "failure_probability": "Fail %",
            "air_temperature_K": "Air K",
            "process_temperature_K": "Proc K",
            "rotational_speed_rpm": "rpm",
            "torque_Nm": "Nm",
            "tool_wear_min": "Wear",
        })
        st.dataframe(recent, hide_index=True, width="stretch", height=360)
    st.subheader("Evidence packet")
    st.caption("Compact JSON consumed by the Decision Support System.")
    st.json(st.session_state.twin.evidence_for_dss())

# --- Auto-advance loop ---------------------------------------------------------
if st.session_state.playing and st.session_state.index < total_rows:
    time.sleep(speed)
    st.rerun()
elif st.session_state.playing and st.session_state.index >= total_rows:
    st.session_state.playing = False
    st.success("Stream complete.")
