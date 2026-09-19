"""
dss_engine.py
==============
Step 2 of the DSS fusion layer.

Combines:
  (a) the NLP evidence layer  -- master_nlp_aspects.csv (per-document aspect
      sentiment: Cost & ROI, Security & Privacy, Scalability & Tech,
      Sustainability & Green), optionally master_ai_topics.csv for topics
  (b) the twin evidence layer -- a StandardEvidence dict from evidence_schema.py

...into one Digital Twin Readiness Index per sector, with an explanation of
*why* the score is what it is (which is the explainability gap your own
literature review slide calls out in existing AI-DSS work).

Sector vocabulary is shared across both layers: "healthcare", "manufacturing",
"business" (this is the retail twin's domain — the NLP pipeline's KEYWORDS
dict in config.py also buckets logistics/supply-chain/retail sources under
"business", so the mapping is already consistent, nothing to rename).
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ASPECT_COLUMNS = [
    "Cost & ROI",
    "Security & Privacy",
    "Scalability & Tech",
    "Sustainability & Green",
]

VALID_SECTORS = ("healthcare", "manufacturing", "business")

# Fusion weights: how much the readiness index trusts the macro (literature/
# news sentiment) signal vs. the live operational (twin) signal. Twin signal
# dominates because it reflects *current* conditions; NLP signal reflects
# general sentiment/context around the technology, which moves more slowly.
# Document and justify this split in your report rather than treating it as
# arbitrary -- the retail twin's business_health_score() weights are a good
# model for how to write that justification.
NLP_WEIGHT = 0.35
TWIN_WEIGHT = 0.65


@dataclass
class NLPEvidence:
    sector: str
    n_documents: int
    aspect_means: dict          # {aspect_name: mean sentiment in [-1, 1] or None}
    overall_sentiment: float    # mean across all aspects with data, in [-1, 1]
    coverage: dict              # {aspect_name: how many docs mentioned it}
    # Every number above is DSS_DERIVED: aspect_sentiment.py's VADER scores
    # are computed per-document from OBSERVED article/abstract text, but the
    # sector-level means and overall_sentiment this class returns are this
    # fusion layer's own aggregation, not something the NLP pipeline itself
    # outputs -- so they're tagged DSS_DERIVED, matching the vocabulary in
    # evidence_schema.py, not passed through as if they were raw model output.
    provenance: str = "DSS_DERIVED"


def load_nlp_evidence(aspects_csv: str, sector: str) -> NLPEvidence:
    """
    Aggregates master_nlp_aspects.csv down to one macro sentiment signal
    per sector. Aspect columns are per-document VADER compound scores
    (see aspect_sentiment.py) in [-1, 1], with NaN where that aspect wasn't
    mentioned in the document -- NaNs are excluded from the mean, not
    treated as zero/neutral, so a sector's score isn't diluted by documents
    that simply didn't discuss (say) sustainability.
    """
    if sector not in VALID_SECTORS:
        raise ValueError(f"sector must be one of {VALID_SECTORS}, got {sector!r}")

    df = pd.read_csv(aspects_csv)
    missing_cols = [c for c in ASPECT_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(f"{aspects_csv} is missing expected columns: {missing_cols}")

    sector_df = df[df["sector"] == sector]
    if sector_df.empty:
        raise ValueError(f"No rows found for sector={sector!r} in {aspects_csv}")

    aspect_means = {}
    coverage = {}
    for aspect in ASPECT_COLUMNS:
        col = sector_df[aspect]
        coverage[aspect] = int(col.notna().sum())
        aspect_means[aspect] = round(float(col.mean()), 4) if col.notna().any() else None

    present = [v for v in aspect_means.values() if v is not None]
    overall_sentiment = round(sum(present) / len(present), 4) if present else 0.0

    return NLPEvidence(
        sector=sector,
        n_documents=len(sector_df),
        aspect_means=aspect_means,
        overall_sentiment=overall_sentiment,
        coverage=coverage,
    )


@dataclass
class ReadinessResult:
    sector: str
    readiness_index: float
    nlp_component: float
    twin_component: float
    status: str
    explanation: list = field(default_factory=list)
    nlp_evidence: NLPEvidence = None
    twin_evidence: dict = None
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    provenance: dict = field(default_factory=lambda: {
        "nlp_component": "DSS_DERIVED",       # aggregated sentiment, see NLPEvidence
        "twin_component": "SEE_TWIN_EVIDENCE",  # defer to twin_evidence['provenance']
        "readiness_index": "DSS_DERIVED",       # this module's own weighted fusion
    })


def compute_readiness_index(nlp_evidence: NLPEvidence, twin_evidence: dict) -> ReadinessResult:
    """
    Fuses one sector's NLP macro signal with its twin's live operational
    signal into a single 0-100 Digital Twin Readiness Index, plus a short
    plain-language explanation of what drove the number -- this is the
    explainability layer your literature review identifies as missing from
    existing AI-DSS work.
    """
    if nlp_evidence.sector != twin_evidence["domain"]:
        raise ValueError(
            f"Sector mismatch: NLP evidence is for {nlp_evidence.sector!r}, "
            f"twin evidence is for {twin_evidence['domain']!r}"
        )

    # NLP macro signal: rescale overall_sentiment from [-1, 1] to [0, 100]
    nlp_component = 50.0 + 50.0 * nlp_evidence.overall_sentiment

    # Twin operational signal: already 0-100 in StandardEvidence
    twin_component = twin_evidence["operational_score"]

    readiness_index = NLP_WEIGHT * nlp_component + TWIN_WEIGHT * twin_component
    readiness_index = round(max(0.0, min(100.0, readiness_index)), 1)

    if readiness_index >= 75:
        status = "Ready"
    elif readiness_index >= 50:
        status = "Developing"
    else:
        status = "At Risk"

    explanation = []
    explanation.append(
        f"Live twin status is '{twin_evidence['status']}' "
        f"(operational score {twin_component:.1f}/100, {twin_evidence['score_provenance']})."
    )
    if twin_evidence["risk_flags"]:
        explanation.append("Twin flags: " + "; ".join(twin_evidence["risk_flags"]))
    sentiment_word = (
        "positive" if nlp_evidence.overall_sentiment > 0.1
        else "negative" if nlp_evidence.overall_sentiment < -0.1
        else "neutral"
    )
    explanation.append(
        f"Literature/news sentiment across {nlp_evidence.n_documents} documents is "
        f"{sentiment_word} (mean {nlp_evidence.overall_sentiment:+.2f})."
    )
    weakest_aspect = min(
        ((k, v) for k, v in nlp_evidence.aspect_means.items() if v is not None),
        key=lambda kv: kv[1], default=None,
    )
    if weakest_aspect and weakest_aspect[1] < 0:
        explanation.append(
            f"Weakest discourse aspect: '{weakest_aspect[0]}' ({weakest_aspect[1]:+.2f})."
        )
    if abs(nlp_component - twin_component) > 30:
        explanation.append(
            "Note: literature sentiment and live operational health disagree sharply "
            "for this sector -- worth investigating before trusting either signal alone."
        )

    return ReadinessResult(
        sector=nlp_evidence.sector,
        readiness_index=readiness_index,
        nlp_component=round(nlp_component, 1),
        twin_component=round(twin_component, 1),
        status=status,
        explanation=explanation,
        nlp_evidence=nlp_evidence,
        twin_evidence=twin_evidence,
    )


def readiness_to_dict(result: ReadinessResult) -> dict:
    """
    Flatten a ReadinessResult into a JSON-serializable dict for the
    dashboard/report. `provenance` at every level is kept explicit rather
    than flattened away, so a reader (or examiner) can trace any number
    back to OBSERVED data, an ASSUMED input, a SIMULATED/PREDICTED twin
    output, or something this fusion layer itself computed (DSS_DERIVED).
    """
    return {
        "sector": result.sector,
        "readiness_index": result.readiness_index,
        "status": result.status,
        "nlp_component": result.nlp_component,
        "twin_component": result.twin_component,
        "explanation": result.explanation,
        "generated_at": result.generated_at,
        "provenance": {
            **result.provenance,
            "twin_component": result.twin_evidence["score_provenance"],
        },
        "nlp_evidence": {
            "n_documents": result.nlp_evidence.n_documents,
            "overall_sentiment": result.nlp_evidence.overall_sentiment,
            "aspect_means": result.nlp_evidence.aspect_means,
            "coverage": result.nlp_evidence.coverage,
            "provenance": result.nlp_evidence.provenance,
        },
        "twin_evidence": result.twin_evidence,   # already carries its own 'provenance' dict
    }


if __name__ == "__main__":
    # Full end-to-end run: real NLP data + real (non-mocked) twins, via
    # twin_runners.py -- no placeholder evidence anywhere in this path.
    import json
    from evidence_schema import (
        adapt_manufacturing_evidence, adapt_healthcare_evidence, adapt_retail_evidence,
    )
    from twin_runners import run_manufacturing_twin, run_healthcare_twin, run_retail_twin

    here = Path(__file__).parent
    aspects_csv = here / "master_nlp_aspects.csv"

    results = {}

    # --- Manufacturing ---
    raw_mfg = run_manufacturing_twin(here / "manufacturing_twin")
    twin_mfg = adapt_manufacturing_evidence(raw_mfg)
    nlp_mfg = load_nlp_evidence(str(aspects_csv), "manufacturing")
    results["manufacturing"] = compute_readiness_index(nlp_mfg, twin_mfg)

    # --- Healthcare ---
    raw_hc_list = run_healthcare_twin(here / "healthcare_twin")
    twin_hc = adapt_healthcare_evidence(raw_hc_list, scenario="BASE")
    nlp_hc = load_nlp_evidence(str(aspects_csv), "healthcare")
    results["healthcare"] = compute_readiness_index(nlp_hc, twin_hc)

    # --- Business / retail ---
    dss_payload, health_score = run_retail_twin(here / "retail_twin")
    twin_biz = adapt_retail_evidence(dss_payload, health_score)
    nlp_biz = load_nlp_evidence(str(aspects_csv), "business")
    results["business"] = compute_readiness_index(nlp_biz, twin_biz)

    for sector, result in results.items():
        print(f"\n{'=' * 20} {sector.upper()} {'=' * 20}")
        print(json.dumps(readiness_to_dict(result), indent=2, default=str))
