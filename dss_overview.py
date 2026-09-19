"""
dss_overview.py
================
Step 4 of the DSS fusion layer: the unified dashboard your Phase 7
("Integrate NLP, DSS, and Digital Twins into a unified dashboard") calls for.

This is the "DSS Overview" page. It is opened through the entry point:
    streamlit run dashboard.py
(dashboard.py owns page config and navigation; each twin's own dashboard is a
separate page, registered in twin_embed.py.)

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
import plotly.graph_objects as go
import streamlit as st

from evidence_schema import (
    adapt_manufacturing_evidence,
    adapt_healthcare_evidence,
    adapt_retail_evidence,
    PROVENANCE_LEVELS,
)
from dss_engine import load_nlp_evidence, compute_readiness_index, readiness_to_dict
from twin_runners import run_manufacturing_twin, run_healthcare_twin, run_retail_twin
from twin_embed import TWINS
import ui_theme
from ui_theme import pill, prov_tag, show_plotly, STATUS_COLORS, ACCENT, NAVY, GOOD, WARN, BAD

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

SECTOR_COLORS = {
    "manufacturing": ACCENT,
    "healthcare": NAVY,
    "business": WARN,
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
ui_theme.inject()

SCORE_PROVENANCE_LABELS = {
    "twin": "TWIN-COMPUTED",
    "dss_derived": "DSS_DERIVED",
}
SCORE_PROVENANCE_COLORS = {
    "twin": ACCENT,
    "dss_derived": ui_theme.INK,
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
    color = SCORE_PROVENANCE_COLORS.get(value, ui_theme.MUTED)
    return prov_tag(label, {label: color})


@st.cache_resource(show_spinner="Running all three Digital Twins and fusing with NLP evidence...")
def get_results():
    return build_all_results()


st.markdown(
    '<div class="eyebrow">FYP · AI-Based DSS for Digital Twin-Driven Transformation</div>',
    unsafe_allow_html=True,
)
st.title("AI-Based Intelligent Decision Support System")
st.caption(
    "Unified view across Healthcare, Manufacturing and Business. "
    "Every number below is traceable — see the provenance legend."
)

brief_l, brief_r = st.columns(2)
with brief_l:
    st.markdown(
        """
        <div class="brief-card">
        <h3>Abstract</h3>
        <p>This project implements an AI-based decision support system that fuses
        literature and news sentiment (NLP) with live digital-twin evidence from
        three domains. Each twin turns operational data into a compact, provenance-tagged
        evidence packet. The fusion layer then produces an explainable
        <b>Digital Twin Readiness Index</b> per sector — so a decision maker can see
        both the technology-context signal and the current operational signal,
        and why the combined score is what it is.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
with brief_r:
    st.markdown(
        """
        <div class="brief-card">
        <h3>Objectives</h3>
        <ol>
          <li>Prototype healthcare, manufacturing and retail digital twins that emit
              decision-ready evidence rather than raw telemetry.</li>
          <li>Aggregate aspect-level NLP sentiment as a macro readiness context signal.</li>
          <li>Fuse NLP + twin evidence into one explainable readiness index per sector.</li>
          <li>Present a single dashboard where scores, graphs and provenance stay readable
              and internally consistent.</li>
        </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

_twin_pages = st.session_state.get("_dss_twin_pages", {})
if _twin_pages:
    st.subheader("Domain digital twins")
    _cols = st.columns(len(_twin_pages))
    for _col, (_key, _page) in zip(_cols, _twin_pages.items()):
        with _col:
            with st.container(border=True):
                st.markdown(f"**{TWINS[_key]['label']}**")
                st.caption(TWINS[_key]["blurb"])
                st.page_link(_page, label=f"Open {TWINS[_key]['label']}",
                             icon=":material/arrow_forward:")

st.markdown(
    "**Provenance:** " + " ".join(prov_tag(p) for p in PROVENANCE_LEVELS),
    unsafe_allow_html=True,
)
st.caption(
    "OBSERVED = raw sensor/dataset value · ASSUMED = a modelling input the twin's "
    "author chose · CALIBRATED = derived from observed data by a documented formula · "
    "SIMULATED = produced by running a twin's state-transition logic forward · "
    "PREDICTED = output of a trained ML model · "
    "DSS_DERIVED = computed only by this fusion layer."
)

if st.button("Re-run all twins"):
    get_results.clear()

results, errors = get_results()

if errors:
    for sector, msg in errors.items():
        st.error(f"**{SECTOR_LABELS.get(sector, sector)}** failed to run: {msg}")

st.header("Digital Twin Readiness Index")

if results:
    card_cols = st.columns(len(results))
    for col, (sector, result) in zip(card_cols, results.items()):
        with col:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="readiness-label">{SECTOR_LABELS[sector]}</div>'
                f'<div class="readiness-score">{result.readiness_index}</div>'
                f'{pill(result.status, STATUS_COLORS)}'
                f'<div style="margin-top:10px;font-size:0.85rem;color:#5C6B7A;">'
                f'Twin {pill(result.twin_evidence["status"], STATUS_COLORS)}'
                f'</div></div>',
                unsafe_allow_html=True,
            )

    chart_l, chart_r = st.columns([1.35, 1])
    with chart_l:
        fig = go.Figure()
        sectors = list(results.keys())
        fig.add_trace(go.Bar(
            name="Readiness index",
            x=[SECTOR_LABELS[s] for s in sectors],
            y=[results[s].readiness_index for s in sectors],
            marker_color=[SECTOR_COLORS[s] for s in sectors],
            text=[f"{results[s].readiness_index:.1f}" for s in sectors],
            textposition="outside",
        ))
        fig.add_hline(y=75, line_dash="dot", line_color=GOOD,
                      annotation_text="Ready ≥ 75", annotation_font_color=GOOD)
        fig.add_hline(y=50, line_dash="dot", line_color=WARN,
                      annotation_text="Developing ≥ 50", annotation_font_color=WARN)
        fig.update_yaxes(range=[0, 110], title="Score (0–100)")
        fig.update_layout(showlegend=False, title="Readiness by sector")
        show_plotly(fig, height=340)
    with chart_r:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="NLP component (35%)",
            x=[SECTOR_LABELS[s] for s in results],
            y=[results[s].nlp_component for s in results],
            marker_color=NAVY,
        ))
        fig.add_trace(go.Bar(
            name="Twin component (65%)",
            x=[SECTOR_LABELS[s] for s in results],
            y=[results[s].twin_component for s in results],
            marker_color=ACCENT,
        ))
        fig.update_layout(barmode="group", title="Fusion components",
                          yaxis_title="Score (0–100)", yaxis_range=[0, 110])
        show_plotly(fig, height=340)
else:
    st.warning("No sector results available — check the errors above and your TWIN_DIRS paths.")

st.header("Sector detail")

if results:
    sector = st.selectbox(
        "Choose a sector", list(results.keys()),
        format_func=lambda s: SECTOR_LABELS[s],
    )
    result = results[sector]
    d = readiness_to_dict(result)

    left, right = st.columns([1, 1.15], gap="large")

    with left:
        st.subheader("Readiness fusion")
        st.markdown(
            f'Readiness Index: **{d["readiness_index"]}** '
            f'{pill(d["status"], STATUS_COLORS)}  '
            f'{prov_tag(d["provenance"]["readiness_index"])}',
            unsafe_allow_html=True,
        )
        st.markdown(f"- NLP component: **{d['nlp_component']:.1f}** "
                    f"{prov_tag(d['provenance']['nlp_component'])}", unsafe_allow_html=True)
        st.markdown(f"- Twin component: **{d['twin_component']:.1f}** "
                    f"{score_prov_tag(d['provenance']['twin_component'])}",
                    unsafe_allow_html=True)

        fuse_fig = go.Figure(go.Bar(
            x=["NLP", "Twin", "Readiness"],
            y=[d["nlp_component"], d["twin_component"], d["readiness_index"]],
            marker_color=[NAVY, ACCENT, SECTOR_COLORS[sector]],
            text=[f"{d['nlp_component']:.1f}", f"{d['twin_component']:.1f}",
                  f"{d['readiness_index']:.1f}"],
            textposition="outside",
        ))
        fuse_fig.update_yaxes(range=[0, 110], title="Score")
        fuse_fig.update_layout(showlegend=False)
        show_plotly(fuse_fig, height=280)

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
            bar_colors = [GOOD if v >= 0 else BAD for v in aspect_df["Mean sentiment"]]
            nlp_fig = go.Figure(go.Bar(
                x=aspect_df["Mean sentiment"],
                y=aspect_df["Aspect"],
                orientation="h",
                marker_color=bar_colors,
                text=[f"{v:+.2f}" for v in aspect_df["Mean sentiment"]],
                textposition="outside",
            ))
            nlp_fig.update_xaxes(range=[-1.05, 1.05], title="Mean VADER sentiment")
            nlp_fig.add_vline(x=0, line_color=ui_theme.LINE)
            nlp_fig.update_layout(showlegend=False)
            show_plotly(nlp_fig, height=300)
            st.dataframe(aspect_df, hide_index=True, width="stretch")
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
    st.dataframe(pd.DataFrame(metrics_rows), hide_index=True, width="stretch")

    with st.expander("Raw twin evidence (untouched, as produced by the twin itself)"):
        st.json(twin["raw_evidence"])

st.caption(
    "This dashboard fuses NLP-derived literature/news sentiment (macro signal) with "
    "each domain's live twin simulation (operational signal) into one explainable "
    "readiness score per sector. Twin-owned scores stay tagged as twin-computed; "
    "anything invented only here is tagged DSS_DERIVED. None of the twins run "
    "against a live production system."
)
