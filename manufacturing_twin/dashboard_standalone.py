"""
Streamlit UI for the MetroPT-3 air-compressor digital twin.

Replays the reference CSV in time order, scores each 10 s bin with the
frozen sparse autoencoder, and shows the anomaly alarm. This is not a
failure prediction and not a remaining-useful-life display.

Run with:  streamlit run dashboard_standalone.py
"""

import sys
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import stream
import streamlit as st
import twin as twin_module
from dt.models.metropt3_sae import PRIMARY_BUDGET
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import ui_theme
from ui_theme import ACCENT, BAD, NAVY, WARN, show_plotly

st.set_page_config(page_title="Manufacturing Digital Twin", layout="wide")
ui_theme.inject()

BUDGETS = ("0.5", "1.0", "2.0")
DEFAULT_START = "2020-06-04 00:00:00"
DEFAULT_END = "2020-06-06 00:00:00"

# Published behaviour of this frozen model. Kept next to the chart so the
# demo cannot be read as a claim of reliable fault prediction.
KNOWN_LIMITS = [
    "The score is a reconstruction error. An alarm means the error crossed a frozen threshold, not that a failure was predicted and not a remaining useful life.",
    "Thresholds were chosen on validation presumed-normal rows for budgets of 0.5, 1.0 and 2.0 episodes per day. The primary budget is 1.0 per day (threshold about 0.02375). Alarms within 30 minutes are one episode.",
    "On the published test period (2020-05-29 to 2020-09-01) budget 1.0 detects reported event E3 only, about 4 seconds after the reported start, with about 20% of rows inside that reported interval above the threshold. It misses E2 and E4. Budget 2.0 also catches E2 and still misses E4, at about 1.9 false-alarm episodes per day. Budget 0.5 detects E3 only, weakly.",
    "Event times are reported labels. They are not verified fault onsets. E1 is day-level only, and E2's maintenance note (30 Apr) does not match its reported interval. The timezone of the log is unknown.",
    "False-alarm episodes per observed day on the test period were about 0.24 (budget 0.5), 0.27 (budget 1.0) and 1.89 (budget 2.0). Overlap with reported events is a lower bound on precision because periods labelled normal were not verified.",
    "The budget 1.0 threshold is sensitive: on validation the alarm rate was about 0.22 episodes/day in the first half and about 1.61 in the second half. Monthly p95 scores fall from about 1e-2 (April–May) to about 1e-4 (July–August).",
    "Train-to-validation drift: DV_pressure standard-deviation ratio about 2.1, and about 4% of TP3 validation values sit outside the training range.",
    "There are four reported events, all air leaks. That supports an anomaly-detection demo on real data. It does not support a claim of reliable fault prediction.",
]


def _load_window(start: str, end: str) -> pd.DataFrame:
    return stream.load_window(start=start, end=end)


st.sidebar.markdown('<div class="eyebrow">Decision Support Console</div>', unsafe_allow_html=True)
st.sidebar.title("Manufacturing Twin")
st.sidebar.caption(
    "Metro air-compressor APU. Replay of MetroPT-3 through a frozen sparse "
    "autoencoder. The banner is an anomaly alarm, not a failure prediction."
)

try:
    stream.resolve_data_path()
    data_ready = True
    data_error = ""
except FileNotFoundError as exc:
    data_ready = False
    data_error = str(exc)

budget = st.sidebar.segmented_control(
    "False-alarm budget (episodes/day)",
    BUDGETS,
    default=PRIMARY_BUDGET,
    help=(
        "Primary budget is 1.0 episode/day. 0.5 is stricter and 2.0 is looser. "
        "Thresholds are frozen from validation and are not refit here."
    ),
)
if budget is None:
    budget = PRIMARY_BUDGET
start_text = st.sidebar.text_input("Window start", DEFAULT_START)
end_text = st.sidebar.text_input("Window end", DEFAULT_END, help="End is exclusive.")
speed = st.sidebar.slider(
    "Playback delay (seconds per step)",
    0.0, 1.0, 0.05, 0.05,
    help="Wall-clock pause between UI steps. Each bin is 10 seconds of compressor time.",
)
step_size = st.sidebar.slider(
    "Bins per step",
    1, 120, 30,
    help="How many already-scored 10 s bins to reveal on each step.",
)

if not data_ready:
    st.title("Manufacturing Digital Twin")
    st.error(data_error)
    st.info(
        "Download MetroPT-3 (UCI dataset 791, file MetroPT3(AirCompressor).csv) "
        "and save it as manufacturing_twin/data/MetroPT3_AirCompressor_.csv. "
        "The model artifacts are already in models/metropt3_sae/."
    )
    with st.expander("What this model can and cannot support"):
        for line in KNOWN_LIMITS:
            st.markdown(f"- {line}")
    st.stop()

window_key = (start_text, end_text)
if st.session_state.get("window_key") != window_key or "twin" not in st.session_state:
    try:
        raw = _load_window(start_text, end_text)
    except (OSError, ValueError) as exc:
        st.error(f"Could not read the replay window: {exc}")
        st.stop()
    if raw.empty:
        st.warning("That window has no rows. Check the start and end timestamps.")
        st.stop()
    machine = twin_module.MachineTwin(budget=budget)
    machine.replay_frame(raw)
    st.session_state.twin = machine
    st.session_state.index = 0
    st.session_state.window_key = window_key
    st.session_state.playing = False
elif st.session_state.twin.budget != budget:
    st.session_state.twin.set_budget(budget)

if "playing" not in st.session_state:
    st.session_state.playing = False

machine = st.session_state.twin
total = len(machine.history)

col_play, col_reset = st.sidebar.columns(2)
if col_play.button("Play" if not st.session_state.playing else "Pause"):
    st.session_state.playing = not st.session_state.playing
if col_reset.button("Reset"):
    st.session_state.index = 0
    st.session_state.playing = False

if st.session_state.playing and st.session_state.index < total:
    st.session_state.index = min(total, st.session_state.index + step_size)

shown = st.session_state.index
st.sidebar.progress(shown / max(total, 1))
st.sidebar.caption(f"Bin {shown} / {total} (each bin is 10 s of data)")

st.markdown(
    '<div class="eyebrow">FYP · AI-Based DSS for Digital Twin-Driven Transformation</div>',
    unsafe_allow_html=True,
)
st.title("Manufacturing Digital Twin")
st.caption(
    "Anomaly detection on the MetroPT-3 air compressor. "
    "The default window includes reported event E3. Reported start times are "
    "dataset labels, not verified fault onsets."
)

with st.expander("What this alarm means — read this before the chart"):
    for line in KNOWN_LIMITS:
        st.markdown(f"- {line}")

if shown == 0:
    st.info("Press Play to walk the scored window. Scoring uses trailing windows only; the first 30 minutes stay unscored.")
    st.stop()

state = machine.history[shown - 1]
hist = machine.history[:shown]
now = state["timestamp"]
alarm = bool(state["alarm"])

if state.get("model_may_be_drifting"):
    st.warning(
        "Model may be drifting: the alarm rate over the last 7 days is more than "
        f"twice the selected budget ({budget} episodes/day)."
    )
elif shown > 1:
    span = pd.Timestamp(now) - pd.Timestamp(hist[0]["timestamp"])
    if span < pd.Timedelta(days=7):
        st.caption(
            "Drift banner stays off until this replay covers 7 days. "
            "The published model does drift: see the note above."
        )

if alarm:
    st.error(state["recommendation"])
elif state["data_quality"]:
    st.info(state["recommendation"])
else:
    st.success(state["recommendation"])

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "Anomaly score (MSE)",
    "—" if state["anomaly_score"] is None else f"{state['anomaly_score']:.5f}",
    help="Mean squared reconstruction error on the 23 standardised features. Not a probability.",
)
k2.metric(
    "Frozen threshold",
    f"{state['threshold']:.5f}",
    help=f"Budget {budget} episodes/day. Chosen on validation and not refit on this replay.",
)
k3.metric(
    "Anomaly alarm",
    "ALARM" if alarm else "no alarm",
    help="True when a finite score is at or above the frozen threshold.",
)
k4.metric(
    "Operating state",
    state["operating_state"] or "—",
    help=(
        "Display label from Motor_current and COMP. "
        "Off if current is below 2 A. Otherwise load if COMP < 0.5 "
        "(COMP is active when the unit is off or offloaded). Not a model output."
    ),
)

st.subheader("Top contributing features")
st.caption("Largest per-feature squared errors on this bin. They explain the reconstruction, not a root cause.")
if state["top_features"]:
    st.write(", ".join(state["top_features"]))
else:
    st.caption("No contribution ranking while the window is incomplete or a sensor is missing.")
if state["data_quality"]:
    st.caption("Data-quality flags: " + ", ".join(state["data_quality"]))

visible_episodes = []
for episode in machine.episodes:
    if episode["end"] <= now:
        visible_episodes.append(episode)
    elif episode["start"] <= now and state.get("active_episode"):
        visible_episodes.append(state["active_episode"])

chart_col, side_col = st.columns([1.6, 1], gap="large")
hist_df = pd.DataFrame(hist)

with chart_col:
    st.subheader("Anomaly score")
    scored = hist_df[hist_df["anomaly_score"].notna()]
    fig = go.Figure()
    if not scored.empty:
        fig.add_trace(go.Scatter(
            x=scored["timestamp"],
            y=scored["anomaly_score"],
            mode="lines",
            name="Reconstruction MSE",
            line={"color": BAD if alarm else ACCENT, "width": 1.6},
            hovertemplate="%{x}<br>MSE %{y:.5f}<extra></extra>",
        ))
    fig.add_hline(
        y=state["threshold"],
        line_dash="dot",
        line_color=WARN,
        annotation_text=f"budget {budget}",
        annotation_font_color=WARN,
    )
    fig.update_layout(
        showlegend=False,
        yaxis_title="Reconstruction MSE",
        xaxis_title="Timestamp (timezone unknown)",
    )
    show_plotly(fig, height=360)

    st.subheader("Sensors in this window")
    sensors = make_subplots(
        rows=2, cols=2,
        subplot_titles=("TP2 (bar)", "TP3 (bar)", "Motor current (A)", "Oil temperature (°C)"),
        vertical_spacing=0.16, horizontal_spacing=0.1,
    )
    x = hist_df["timestamp"]
    sensors.add_trace(go.Scatter(x=x, y=hist_df["tp2_bar"], name="TP2",
                                 line={"color": NAVY, "width": 1.3}), row=1, col=1)
    sensors.add_trace(go.Scatter(x=x, y=hist_df["tp3_bar"], name="TP3",
                                 line={"color": ACCENT, "width": 1.3}, showlegend=False), row=1, col=2)
    sensors.add_trace(go.Scatter(x=x, y=hist_df["motor_current_A"], name="Current",
                                 line={"color": WARN, "width": 1.3}, showlegend=False), row=2, col=1)
    sensors.add_trace(go.Scatter(x=x, y=hist_df["oil_temperature_C"], name="Oil",
                                 line={"color": BAD, "width": 1.3}, showlegend=False), row=2, col=2)
    sensors.update_layout(showlegend=False)
    show_plotly(sensors, height=420)

with side_col:
    st.subheader("Alarm episodes")
    st.caption("Alarms within 30 minutes are one episode. Start times below are when the score crossed the threshold.")
    if visible_episodes:
        ep_df = pd.DataFrame(visible_episodes)[
            ["start", "end", "n_rows", "peak_score", "top_features", "budget"]
        ]
        ep_df["top_features"] = ep_df["top_features"].apply(
            lambda names: ", ".join(names) if isinstance(names, list) else names
        )
        st.dataframe(ep_df.iloc[::-1], hide_index=True, width="stretch", height=280)
    else:
        st.caption("No episode has opened in the revealed part of the window.")

    if state.get("active_episode"):
        st.markdown("**Active episode**")
        st.json(state["active_episode"])

    st.subheader("Evidence packet")
    st.caption("What the DSS receives for the revealed bin. There is no failure probability and no tool-wear KPI.")
    # evidence_for_dss reads latest_state, which is the end of the loaded window.
    # Show the revealed bin instead, so playback does not leak the future.
    st.json({
        "domain": "Manufacturing",
        "machine": state["machine"],
        "timestamp": state["timestamp"],
        "alarm": state["alarm"],
        "anomaly_score": state["anomaly_score"],
        "threshold": state["threshold"],
        "budget": state["budget"],
        "operating_state": state["operating_state"],
        "top_features": state["top_features"],
        "active_episode": state["active_episode"],
        "data_quality": state["data_quality"],
        "model_may_be_drifting": state["model_may_be_drifting"],
        "recommendation": state["recommendation"],
        "score_kind": "reconstruction_mse",
    })

if st.session_state.playing and shown < total:
    time.sleep(speed)
    st.rerun()
elif st.session_state.playing and shown >= total:
    st.session_state.playing = False
    st.success("Replay window complete.")
