"""
Shared visual language for the unified DSS dashboard and the three twins.

Keep tokens, Plotly styling, and chrome CSS in one place so the overview
page and the domain twins read as one application rather than four
independently themed prototypes.
"""

from __future__ import annotations

INK = "#1B2430"
MUTED = "#5C6B7A"
BG = "#F3F5F8"
PANEL = "#FFFFFF"
LINE = "#E2E8EF"
ACCENT = "#1F6B6B"
ACCENT_SOFT = "#E6F2F1"
NAVY = "#2F5D8C"
NAVY_SOFT = "#E8F0F8"
GOOD = "#2F8A5A"
WARN = "#C47A2C"
BAD = "#B5453A"
VIOLET = "#6B5B95"

STATUS_COLORS = {
    "Healthy": GOOD,
    "Warning": WARN,
    "Critical": BAD,
    "Ready": GOOD,
    "Developing": WARN,
    "At Risk": BAD,
}

PROVENANCE_COLORS = {
    "OBSERVED": ACCENT,
    "ASSUMED": VIOLET,
    "CALIBRATED": NAVY,
    "SIMULATED": WARN,
    "PREDICTED": BAD,
    "DSS_DERIVED": INK,
}

PLOTLY_FONT = dict(family="IBM Plex Sans, Source Sans 3, system-ui, sans-serif",
                   color=INK, size=12)

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500&display=swap');

:root {{
    --bg: {BG};
    --panel: {PANEL};
    --line: {LINE};
    --text: {INK};
    --sub: {MUTED};
    --accent: {ACCENT};
    --accent-soft: {ACCENT_SOFT};
    --navy: {NAVY};
    --navy-soft: {NAVY_SOFT};
    --amber: {WARN};
    --amber-light: #FBF3E6;
    --good: {GOOD};
    --bad: {BAD};
}}

html, body, .stApp, [data-testid="stAppViewContainer"] {{
    background-color: var(--bg) !important;
    color: var(--text);
    font-family: "IBM Plex Sans", "Source Sans 3", system-ui, sans-serif;
}}
section[data-testid="stSidebar"] {{
    background-color: var(--panel) !important;
    border-right: 1px solid var(--line);
}}
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stToolbar"] {{ background: transparent; }}

h1, h2, h3, h4 {{
    color: var(--text) !important;
    font-family: "IBM Plex Sans", system-ui, sans-serif !important;
    font-weight: 650 !important;
    letter-spacing: -0.02em;
}}
h1 {{ font-size: 1.85rem !important; }}
h2 {{ font-size: 1.28rem !important; margin-top: 0.4rem; }}
h3 {{ font-size: 1.08rem !important; }}

[data-testid="stMetric"] {{
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 12px;
    padding: 12px 14px 10px;
    box-shadow: 0 1px 2px rgba(27, 36, 48, 0.04);
}}
[data-testid="stMetricLabel"] {{ color: var(--sub) !important; }}
[data-testid="stMetricValue"] {{ color: var(--text) !important; }}

div[data-testid="stTabs"] [data-baseweb="tab-list"] {{
    gap: 4px;
    background: transparent;
    border-bottom: 1px solid var(--line);
}}
div[data-testid="stTabs"] [data-baseweb="tab"] {{
    color: var(--sub);
    font-weight: 550;
}}
div[data-testid="stTabs"] [aria-selected="true"] {{
    color: var(--accent) !important;
}}

.stButton > button, .stDownloadButton > button {{
    background-color: {PANEL} !important;
    color: {INK} !important;
    border: 1px solid #C9D2DC !important;
    border-radius: 8px !important;
}}
.stButton > button:hover, .stDownloadButton > button:hover {{
    color: {ACCENT} !important;
    border-color: {ACCENT} !important;
}}
.stButton > button[kind="primary"],
.stButton > button[data-testid="stBaseButton-primary"] {{
    background-color: {ACCENT} !important;
    color: #FFFFFF !important;
    border: none !important;
}}

.eyebrow {{
    font-family: "IBM Plex Mono", ui-monospace, monospace;
    font-size: 11px;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--accent);
    margin-bottom: 2px;
}}
.hero-card, .metric-card, .brief-card {{
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 16px 18px;
    box-shadow: 0 1px 2px rgba(27, 36, 48, 0.04);
}}
.brief-card h3 {{
    margin: 0 0 8px 0;
    font-size: 0.95rem !important;
    color: var(--accent) !important;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}}
.brief-card p, .brief-card li {{
    color: var(--text);
    font-size: 0.92rem;
    line-height: 1.55;
    margin: 0;
}}
.brief-card ol {{ margin: 0; padding-left: 1.15rem; }}
.brief-card li {{ margin-bottom: 0.35rem; }}

.status-pill, .prov-tag {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 0.75rem;
    font-weight: 650;
    color: #fff;
    letter-spacing: .02em;
}}
.prov-tag {{ margin-right: 4px; }}

.readiness-score {{
    font-size: 2.4rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1.1;
    color: var(--text);
}}
.readiness-label {{
    font-size: 0.78rem;
    color: var(--sub);
    letter-spacing: .08em;
    text-transform: uppercase;
    margin-bottom: 4px;
}}

.disclaimer, .note, .warn {{
    font-size: 12.5px;
    padding: 10px 14px;
    border-radius: 10px;
    line-height: 1.5;
    margin-bottom: 12px;
}}
.disclaimer {{
    border: 1px solid #C5D9D8;
    background: var(--accent-soft);
    color: #134848;
}}
.note {{
    border: 1px solid var(--line);
    background: #FAFBFD;
    color: var(--sub);
}}
.warn {{
    border: 1px solid #F3D0A0;
    background: var(--amber-light);
    color: var(--amber);
}}

.legend-row {{
    display: flex; gap: 10px; margin-bottom: 16px; flex-wrap: wrap;
}}
.legend-chip {{
    flex: 1; min-width: 200px;
    border-radius: 10px; padding: 9px 12px;
    font-size: 12px; line-height: 1.45;
    border: 1px solid var(--line); background: var(--panel);
}}
.legend-chip b {{
    display: block; font-size: 11px; letter-spacing: .04em;
    text-transform: uppercase; margin-bottom: 3px;
}}
.legend-observed b {{ color: {ACCENT}; }}
.legend-assumed b {{ color: {VIOLET}; }}
.legend-derived b {{ color: {NAVY}; }}
.legend-simulated b {{ color: {WARN}; }}

.health-banner {{
    padding: 1rem 1.1rem;
    border-radius: 12px;
    text-align: center;
    border: 1.5px solid;
}}
.health-banner .k {{
    font-size: 0.72rem;
    letter-spacing: .12em;
    text-transform: uppercase;
    color: var(--sub);
}}
.health-banner .v {{
    font-size: 1.7rem;
    font-weight: 700;
    margin-top: 2px;
}}
</style>
"""


def inject() -> None:
    import streamlit as st
    st.markdown(CSS, unsafe_allow_html=True)


def pill(label: str, color_map: dict | None = None) -> str:
    color = (color_map or STATUS_COLORS).get(label, MUTED)
    return f'<span class="status-pill" style="background:{color}">{label}</span>'


def prov_tag(label: str, color_map: dict | None = None) -> str:
    color = (color_map or PROVENANCE_COLORS).get(label, MUTED)
    return f'<span class="prov-tag" style="background:{color}">{label}</span>'


def style_plotly(fig, height: int = 360, time_axis: bool = False):
    """Light, high-contrast Plotly layout used by every dashboard chart."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font=PLOTLY_FONT,
        margin=dict(l=16, r=16, t=36, b=16),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            bgcolor="rgba(255,255,255,0.8)",
            font=dict(color=INK, size=12),
        ),
        hoverlabel=dict(bgcolor=PANEL, bordercolor=LINE, font=dict(color=INK)),
        colorway=[ACCENT, NAVY, WARN, GOOD, BAD, VIOLET],
    )
    fig.update_xaxes(
        gridcolor="#EEF2F6",
        zerolinecolor=LINE,
        color=INK,
        tickfont=dict(color=INK),
        title_font=dict(color=MUTED, size=12),
        showline=True,
        linecolor=LINE,
    )
    fig.update_yaxes(
        gridcolor="#EEF2F6",
        zerolinecolor=LINE,
        color=INK,
        tickfont=dict(color=INK),
        title_font=dict(color=MUTED, size=12),
        showline=True,
        linecolor=LINE,
    )
    if time_axis:
        fig.update_layout(margin=dict(l=16, r=16, t=48, b=16))
    return fig


def show_plotly(fig, height: int = 360, time_axis: bool = False) -> None:
    import streamlit as st
    style_plotly(fig, height=height, time_axis=time_axis)
    st.plotly_chart(fig, use_container_width=True, theme=None)
