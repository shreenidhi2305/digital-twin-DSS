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

import time

import pandas as pd
import streamlit as st

import stream
import twin as twin_module

st.set_page_config(page_title="Manufacturing Digital Twin", layout="wide")

HEALTH_COLORS = {
    "Healthy": "#2ecc71",
    "Warning": "#f39c12",
    "Critical": "#e74c3c",
}

# --- Session state: persists the twin + stream position across reruns --
if "twin" not in st.session_state:
    st.session_state.twin = twin_module.MachineTwin("CNC-01")
if "rows" not in st.session_state:
    st.session_state.rows = stream.load_stream().to_dict(orient="records")
if "index" not in st.session_state:
    st.session_state.index = 0
if "playing" not in st.session_state:
    st.session_state.playing = False

total_rows = len(st.session_state.rows)

# --- Sidebar controls ----------------------------------------------------
st.sidebar.title("Digital Twin Controls")
st.sidebar.caption("Manufacturing domain — evidence source for the DSS")

machine_id = st.sidebar.text_input("Machine ID", value=st.session_state.twin.machine_id)
if machine_id != st.session_state.twin.machine_id:
    st.session_state.twin.machine_id = machine_id

speed = st.sidebar.slider("Playback delay (seconds/reading)", 0.0, 1.0, 0.1, 0.05)
step_size = st.sidebar.slider("Readings per step", 1, 50, 1)

col_play, col_reset = st.sidebar.columns(2)
if col_play.button("▶ Play" if not st.session_state.playing else "⏸ Pause"):
    st.session_state.playing = not st.session_state.playing
if col_reset.button("⟲ Reset"):
    st.session_state.twin = twin_module.MachineTwin(machine_id)
    st.session_state.index = 0
    st.session_state.playing = False

st.sidebar.progress(min(st.session_state.index / total_rows, 1.0))
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
st.title("🏭 Manufacturing Digital Twin")
st.caption("Lightweight prototype — evidence source for the Digital Twin Readiness Index / DSS")

if state is None:
    st.info("No readings processed yet. Click ▶ Play or Reset to start the stream.")
    st.stop()

# --- Top-line status ---------------------------------------------------------
health = state["health"]
color = HEALTH_COLORS.get(health, "#95a5a6")

status_col, prob_col, machine_col = st.columns(3)
with status_col:
    st.markdown(
        f"""
        <div style="padding:1rem;border-radius:0.5rem;background:{color}22;
                    border:2px solid {color};text-align:center;">
            <div style="font-size:0.9rem;color:#666;">MACHINE HEALTH</div>
            <div style="font-size:2rem;font-weight:700;color:{color};">{health}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with prob_col:
    st.metric("Failure Probability", f"{state['failure_probability']*100:.2f}%")
with machine_col:
    st.metric("Machine", state["machine"], help=f"Type: {state['machine_type']}")

if health != "Healthy":
    st.warning(f"**Recommendation:** {state['recommendation']}")
else:
    st.success(f"**Recommendation:** {state['recommendation']}")

# --- Operating conditions ---------------------------------------------------
st.subheader("Current Operating Conditions")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Air Temperature", f"{state['air_temperature_K']} K")
c2.metric("Process Temperature", f"{state['process_temperature_K']} K")
c3.metric("Rotational Speed", f"{state['rotational_speed_rpm']} rpm")
c4.metric("Torque", f"{state['torque_Nm']} Nm")
st.metric("Tool Wear", f"{state['tool_wear_min']} min")

# --- History chart -----------------------------------------------------------
st.subheader("Failure Probability Over Time")
hist_df = pd.DataFrame(st.session_state.twin.history)
if not hist_df.empty:
    chart_df = hist_df[["failure_probability"]].reset_index(drop=True)
    st.line_chart(chart_df, height=250)

# --- Recent readings table ----------------------------------------------------
with st.expander("Recent readings"):
    st.dataframe(
        hist_df.tail(20)[
            ["timestamp", "health", "failure_probability", "air_temperature_K",
             "process_temperature_K", "rotational_speed_rpm", "torque_Nm", "tool_wear_min"]
        ].iloc[::-1],
        use_container_width=True,
    )

# --- DSS evidence packet -------------------------------------------------------
st.subheader("Evidence Packet Sent to the DSS")
st.caption("This compact JSON — not raw telemetry — is what the Decision Support System consumes.")
st.json(st.session_state.twin.evidence_for_dss())

# --- Auto-advance loop ---------------------------------------------------------
if st.session_state.playing and st.session_state.index < total_rows:
    time.sleep(speed)
    st.rerun()
elif st.session_state.playing and st.session_state.index >= total_rows:
    st.session_state.playing = False
    st.success("Stream complete.")