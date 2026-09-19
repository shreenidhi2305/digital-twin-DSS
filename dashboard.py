"""
dashboard.py
=============
Step 4 of the DSS fusion layer: the unified dashboard your Phase 7
("Integrate NLP, DSS, and Digital Twins into a unified dashboard") calls for.

Run:
    streamlit run dashboard.py

Expects, in the same directory as this file (adjust TWIN_DIRS / ASPECTS_CSV
below if your layout differs):
    manufacturing_twin/   -- the manufacturing_twin project folder
    healthcare_twin/      -- the twin_app project folder
    retail_twin/          -- the retail_digital_twin project folder
    master_nlp_aspects.csv

All computation logic lives in build_all_results() / build_sector_result(),
which have NO Streamlit dependency -- they're plain functions you can also
call from a script or a test. Everything below that point in this file is
UI-only, calling into evidence_schema.py, dss_engine.py and twin_runners.py
(Steps 1-3) rather than recomputing anything itself. That keeps the fusion
logic in one place: if you change how a score is computed, you change it in
dss_engine.py / evidence_schema.py, not here.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from evidence_schema import (
    adapt_manufacturing_evidence,
    adapt_healthcare_evidence,
    adapt_retail_evidence,
    PROVENANCE_LEVELS,
)
from dss_engine import load_nlp_evidence, compute_readiness_index, readiness_to_dict
from twin_runners import run_manufacturing_twin, run_healthcare_twin, run_retail_twin

HERE = Path(__file__).parent
ASPECTS_CSV = HERE / "master_nlp_aspects.csv"
TWIN_DIRS = {
    "manufacturing": HERE / "manufacturing_twin",
    "healthcare": HERE / "healthcare_twin",
    "business": HERE / "retail_twin",
}

SECTOR_LABELS = {
    "manufacturing": "Manufacturing",
    "healthcare": "Healthcare",
    "business": "Business / Retail",
}

STATUS_COLORS = {
    "Healthy": "#3E8E5C", "Warning": "#C77B3B", "Critical": "#B5453A",
    "Ready": "#3E8E5C", "Developing": "#C77B3B", "At Risk": "#B5453A",
}

PROVENANCE_COLORS = {
    "OBSERVED": "#2E6F6E",
    "ASSUMED": "#8A7DAE",
    "CALIBRATED": "#4C7CB0",
    "SIMULATED": "#C77B3B",
    "PREDICTED": "#B5453A",
    "DSS_DERIVED": "#22282B",
}


# ---------------------------------------------------------------------------
# Pure computation -- no Streamlit calls, so this half of the file is
# independently testable and matches exactly what dss_engine.py's own
# __main__ smoke test does (Step 2/3), just packaged per-sector.
# ---------------------------------------------------------------------------
def build_sector_result(sector: str, twin_dirs: dict, aspects_csv: Path):
    """Runs one sector's real twin + real NLP aggregation and fuses them."""
    nlp_evidence = load_nlp_evidence(str(aspects_csv), sector)

    if sector == "manufacturing":
        raw = run_manufacturing_twin(twin_dirs["manufacturing"])
        twin_evidence = adapt_manufacturing_evidence(raw)
    elif sector == "healthcare":
        raw_list = run_healthcare_twin(twin_dirs["healthcare"])
        twin_evidence = adapt_healthcare_evidence(raw_list, scenario="BASE")
    elif sector == "business":
        dss_payload, health_score = run_retail_twin(twin_dirs["business"])
        twin_evidence = adapt_retail_evidence(dss_payload, health_score)
    else:
        raise ValueError(f"Unknown sector: {sector!r}")

    return compute_readiness_index(nlp_evidence, twin_evidence)


def build_all_results(twin_dirs: dict = TWIN_DIRS, aspects_csv: Path = ASPECTS_CSV) -> dict:
    """Runs all three sectors. Returns {sector: ReadinessResult}."""
    results = {}
    errors = {}
    for sector in ("manufacturing", "healthcare", "business"):
        try:
            results[sector] = build_sector_result(sector, twin_dirs, aspects_csv)
        except Exception as exc:  # noqa: BLE001 -- surfaced in the UI, not swallowed
            errors[sector] = str(exc)
    return results, errors


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Digital Twin DSS", page_icon="\U0001F9ED", layout="wide")

st.markdown("""
<style>
    :root {
        --page-bg: #F7F6F2;
        --panel-bg: #FFFFFF;
        --text: #22282B;
        --muted-text: #4F5A5F;
        --border: #E4E1D8;
    }
    .stApp { background-color: var(--page-bg); color: var(--text); }
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Georgia', serif;
        color: var(--text) !important;
    }
    .stMarkdown, .stMarkdown p, .stMarkdown li,
    [data-testid="stCaptionContainer"],
    [data-testid="stWidgetLabel"],
    [data-testid="stText"] {
        color: var(--text);
        line-height: 1.5;
    }
    [data-testid="stCaptionContainer"] { color: var(--muted-text) !important; }
    [data-testid="stWidgetLabel"] p { color: var(--text) !important; }
    [data-testid="stDataFrame"] { color: var(--text); }
    .stButton > button, .stSelectbox label { color: var(--text) !important; }
    .status-pill {
        display: inline-block; padding: 3px 12px; border-radius: 12px;
        font-size: 0.85rem; font-weight: 600; color: white;
    }
    .prov-tag {
        display: inline-block; padding: 1px 8px; border-radius: 8px;
        font-size: 0.7rem; font-weight: 600; color: white; margin-right: 4px;
    }
    .metric-card {
        background: var(--panel-bg); color: var(--text); border-radius: 10px;
        padding: 16px 20px; border: 1px solid var(--border);
        box-shadow: 0 1px 2px rgba(0,0,0,0.03);
    }
    .metric-card div { color: var(--text); }
</style>
""", unsafe_allow_html=True)


def pill(label, color_map):
    color = color_map.get(label, "#888")
    return f'<span class="status-pill" style="background:{color}">{label}</span>'


def prov_tag(label):
    color = PROVENANCE_COLORS.get(label, "#888")
    return f'<span class="prov-tag" style="background:{color}">{label}</span>'


SCORE_PROVENANCE_LABELS = {
    "twin": "TWIN-COMPUTED",
    "dss_derived": "DSS_DERIVED",
}
SCORE_PROVENANCE_COLORS = {
    "twin": "#2E6F6E",
    "dss_derived": PROVENANCE_COLORS["DSS_DERIVED"],
}


def score_prov_tag(value):
    """
    score_provenance uses its own two-value vocabulary ("twin" / "dss_derived")
    -- who computed this particular 0-100 score -- which is a different axis
    from the six-level OBSERVED/ASSUMED/.../DSS_DERIVED vocabulary used for
    individual key_metrics fields. Keeping them as two distinct tag styles
    avoids mislabeling a twin-computed score with a field-provenance tag
    (or vice versa).
    """
    label = SCORE_PROVENANCE_LABELS.get(value, value.upper())
    color = SCORE_PROVENANCE_COLORS.get(value, "#888")
    return f'<span class="prov-tag" style="background:{color}">{label}</span>'


@st.cache_resource(show_spinner="Running all three Digital Twins and fusing with NLP evidence...")
def get_results():
    return build_all_results()


st.title("\U0001F9ED AI-Based Intelligent Decision Support System")
st.caption(
    "Digital Twin-Driven Digital Transformation — unified view across Healthcare, "
    "Manufacturing and Business. Every number below is traceable: see the "
    "provenance legend."
)

# Provenance legend, matching the healthcare twin's own "Observed / Assumed /
# Calibrated-Derived / Simulated" legend convention, extended with the two
# tags this fusion layer adds (PREDICTED for trained-model output, DSS_DERIVED
# for anything computed only here).
st.markdown(
    "**Provenance:** " + " ".join(prov_tag(p) for p in PROVENANCE_LEVELS),
    unsafe_allow_html=True,
)
st.caption(
    "OBSERVED = raw sensor/dataset value · ASSUMED = a modelling input the twin's "
    "author chose · CALIBRATED = derived from observed data by a documented formula · "
    "SIMULATED = produced by running a twin's state-transition logic forward, does not "
    "exist in source data · PREDICTED = output of a trained ML model · "
    "DSS_DERIVED = computed only by this fusion layer, not claimed by any twin."
)

if st.button("\U0001F504 Re-run all twins"):
    get_results.clear()

results, errors = get_results()

if errors:
    for sector, msg in errors.items():
        st.error(f"**{SECTOR_LABELS.get(sector, sector)}** failed to run: {msg}")

st.markdown("---")
st.header("Overview — Digital Twin Readiness Index")

if results:
    cols = st.columns(len(results))
    for col, (sector, result) in zip(cols, results.items()):
        with col:
            st.markdown(f"#### {SECTOR_LABELS[sector]}")
            st.markdown(
                f'<div class="metric-card">'
                f'<div style="font-size:2.2rem; font-weight:700;">{result.readiness_index}</div>'
                f'{pill(result.status, STATUS_COLORS)}'
                f'<div style="margin-top:8px; font-size:0.85rem; color:#555;">'
                f'Twin: {pill(result.twin_evidence["status"], STATUS_COLORS)}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
else:
    st.warning("No sector results available — check the errors above and your TWIN_DIRS paths.")

st.markdown("---")
st.header("Sector Detail")

if results:
    sector = st.selectbox(
        "Choose a sector", list(results.keys()),
        format_func=lambda s: SECTOR_LABELS[s],
    )
    result = results[sector]
    d = readiness_to_dict(result)

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Readiness fusion")
        st.markdown(
            f'Readiness Index: **{d["readiness_index"]}** '
            f'{pill(d["status"], STATUS_COLORS)}  '
            f'{prov_tag(d["provenance"]["readiness_index"])}',
            unsafe_allow_html=True,
        )
        st.markdown(f"- NLP component: **{d['nlp_component']}** "
                    f"{prov_tag(d['provenance']['nlp_component'])}", unsafe_allow_html=True)
        st.markdown(f"- Twin component: **{d['twin_component']}** "
                    f"{score_prov_tag(d['provenance']['twin_component'])}",
                    unsafe_allow_html=True)

        st.subheader("Why this score")
        for line in d["explanation"]:
            st.write(f"- {line}")

    with right:
        st.subheader("NLP evidence (literature + news)")
        nlp = d["nlp_evidence"]
        st.caption(f"{nlp['n_documents']} documents — sector: {sector}  "
                   f"{prov_tag(nlp['provenance'])}", unsafe_allow_html=True)
        aspect_df = pd.DataFrame([
            {"Aspect": k, "Mean sentiment": v, "Documents mentioning it": nlp["coverage"][k]}
            for k, v in nlp["aspect_means"].items() if v is not None
        ])
        if not aspect_df.empty:
            st.bar_chart(aspect_df.set_index("Aspect")["Mean sentiment"])
            st.dataframe(aspect_df, hide_index=True, width='stretch')
        else:
            st.info("No aspect-sentiment data available for this sector.")

    st.subheader("Live twin evidence")
    twin = d["twin_evidence"]
    st.markdown(
        f'Entity: **{twin["entity"]}** — status {pill(twin["status"], STATUS_COLORS)} '
        f'(score {twin["operational_score"]}, {score_prov_tag(twin["score_provenance"])})',
        unsafe_allow_html=True,
    )
    if twin["risk_flags"]:
        for flag in twin["risk_flags"]:
            st.warning(flag)
    st.markdown(f"**Recommendation:** {twin['recommendation']}")

    metrics_rows = []
    for key, value in twin["key_metrics"].items():
        prov = twin["provenance"].get(key, "—")
        metrics_rows.append({"Metric": key, "Value": str(value), "Provenance": prov})
    st.dataframe(pd.DataFrame(metrics_rows), hide_index=True, width='stretch')

    with st.expander("Raw twin evidence (untouched, as produced by the twin itself)"):
        st.json(twin["raw_evidence"])

st.markdown("---")
st.caption(
    "This dashboard fuses NLP-derived literature/news sentiment (macro signal) with "
    "each domain's live twin simulation (operational signal) into one explainable "
    "readiness score per sector — see dss_engine.py for the weighting and its "
    "justification. No twin's own no-composite-score design decision is overridden: "
    "any score not already produced by a twin is explicitly tagged DSS_DERIVED above. "
    "None of the twins here run against a live production system — see each twin's "
    "own README for what is/isn't claimed."
)
