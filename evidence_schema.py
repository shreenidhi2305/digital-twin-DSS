"""
evidence_schema.py
====================
Step 1 of the DSS fusion layer.

PROVENANCE — carried through strictly, matching each twin's own README
vocabulary (the healthcare and retail twins already label every number as
OBSERVED / ASSUMED / CALIBRATED-DERIVED / SIMULATED / PREDICTED; this module
extends that same vocabulary to the fused layer instead of flattening it
away). Every field in `key_metrics` gets a matching entry in `provenance`
using this shared vocabulary:

  OBSERVED        -- read directly from a real dataset, unmodified
  ASSUMED         -- a modelling input the twin's author chose (e.g. a
                     scenario starting condition, a literature-baseline
                     parameter) -- not measured, not fitted
  CALIBRATED       -- fitted/derived from observed data by the twin's own
                     deterministic formula (not a trained ML model)
  SIMULATED        -- produced by running the twin's state-transition /
                     simulation logic forward; does not exist in source data
  PREDICTED        -- output of a trained ML model (classifier/regressor)
  DSS_DERIVED      -- computed by this fusion layer itself (e.g. the
                     healthcare operational_score, which no twin computes on
                     its own — see each README: "that's the overarching
                     DSS's job")

score_provenance on the top-level record uses "twin" (the number/formula is
the twin's own, e.g. PREDICTED or CALIBRATED) or "dss_derived" (computed
here, not by the twin).

Your three twins already each produce a "decision-ready evidence packet"
(manufacturing: twin.evidence_for_dss(), healthcare: dss_interface.build_evidence(),
retail: simulation.export_dss_evidence() payload). They're all well-formed, but
each has a *different shape* and a different notion of "how healthy is this
domain right now" (or, for healthcare, no composite score at all — deliberately,
per its README: "that's the overarching DSS's job").

This module is that translation step: three adapter functions that take each
twin's native output and produce one shared `StandardEvidence` shape. Nothing
here re-simulates or overrides the twins' own numbers — every field in
`raw_evidence` is untouched, and any score this module *does* compute is
clearly labelled as DSS-DERIVED, not as something the twin claimed.

Common schema (a plain dict, so it's trivial to json.dumps / put in a
DataFrame / feed to a template):

{
    "domain":            "manufacturing" | "healthcare" | "business",
    "entity":            str   -- e.g. "CNC-01", "Kamla Nehru Hospital", "retail_network"
    "generated_at":      ISO-8601 str
    "status":            "Healthy" | "Warning" | "Critical"   -- 3-tier, common across domains
    "operational_score":  float 0-100, higher = healthier      -- DSS-derived if the twin
                                                                    didn't already supply one
    "score_provenance":  "twin" | "dss_derived"                -- who computed operational_score
    "risk_flags":        list[str]                              -- short, human-readable
    "recommendation":    str
    "key_metrics":       dict                                   -- the handful of numbers
                                                                    worth surfacing on a card
    "raw_evidence":      dict                                   -- the twin's untouched output,
                                                                    kept for traceability
}
"""

from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Shared 3-tier status vocabulary. Each adapter maps its domain's own
# labels/thresholds onto this so the dashboard can use one status pill design
# everywhere instead of three different vocabularies.
# ---------------------------------------------------------------------------
STATUS_LEVELS = ("Healthy", "Warning", "Critical")


def _now():
    return datetime.now(timezone.utc).isoformat()


PROVENANCE_LEVELS = ("OBSERVED", "ASSUMED", "CALIBRATED", "SIMULATED", "PREDICTED", "DSS_DERIVED")


def _standard(domain, entity, status, operational_score, score_provenance,
              risk_flags, recommendation, key_metrics, raw_evidence, provenance):
    if status not in STATUS_LEVELS:
        raise ValueError(f"status must be one of {STATUS_LEVELS}, got {status!r}")
    bad_tags = {k: v for k, v in provenance.items() if v not in PROVENANCE_LEVELS}
    if bad_tags:
        raise ValueError(f"Unknown provenance tag(s): {bad_tags}; must be one of {PROVENANCE_LEVELS}")
    untagged = set(key_metrics) - set(provenance)
    if untagged:
        raise ValueError(f"key_metrics missing a provenance tag: {untagged}")
    return {
        "domain": domain,
        "entity": entity,
        "generated_at": _now(),
        "status": status,
        "operational_score": round(float(operational_score), 1),
        "score_provenance": score_provenance,
        "risk_flags": risk_flags,
        "recommendation": recommendation,
        "key_metrics": key_metrics,
        "provenance": provenance,
        "raw_evidence": raw_evidence,
    }


# ---------------------------------------------------------------------------
# Manufacturing — from manufacturing_twin/twin.py: MachineTwin.evidence_for_dss()
#
# Native shape:
# {
#   "domain": "Manufacturing", "machine": "CNC-01", "health": "Healthy"|"Warning"|"Critical",
#   "failure_probability": float, "tool_wear": int, "recommendation": str
# }
#
# Already has a 3-tier status and an explicit recommendation, so this adapter
# mostly relabels fields and derives operational_score from failure_probability
# (score_provenance = "twin", since the underlying number — failure_probability —
# is the twin's own model output; we're just rescaling it to 0-100).
# ---------------------------------------------------------------------------
def adapt_manufacturing_evidence(evidence: dict) -> dict:
    if not evidence:
        raise ValueError("Empty manufacturing evidence — has the twin been update()'d yet?")

    health = evidence["health"]
    failure_prob = evidence["failure_probability"]
    tool_wear = evidence.get("tool_wear")

    operational_score = 100.0 * (1.0 - failure_prob)

    risk_flags = []
    if health != "Healthy":
        risk_flags.append(f"Failure probability {failure_prob:.0%} ({health})")
    if tool_wear is not None and tool_wear > 200:
        risk_flags.append(f"Tool wear elevated ({tool_wear} min)")

    return _standard(
        domain="manufacturing",
        entity=evidence.get("machine", "unknown_machine"),
        status=health,
        operational_score=operational_score,
        score_provenance="twin",
        risk_flags=risk_flags,
        recommendation=evidence["recommendation"],
        key_metrics={
            "failure_probability": failure_prob,
            "tool_wear_min": tool_wear,
        },
        provenance={
            # failure_probability is the trained XGBClassifier's own output
            "failure_probability": "PREDICTED",
            # tool_wear is a raw sensor reading, passed through unmodified
            "tool_wear_min": "OBSERVED",
        },
        raw_evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Healthcare — from twin_app/dss_interface.py: build_evidence(profile, summaries, forecast)
#
# Native shape: a LIST of one dict per scenario (LOW/BASE/HIGH/...), each with
# peak/mean occupancy, breach stats, staff pressure, risk_distribution, etc.
# No composite score is computed upstream — deliberately (see its README).
#
# This adapter picks one scenario (default "BASE") to represent current
# operating conditions and *derives* a score here, transparently, from the
# same RISK_THRESHOLDS-style signals the twin already exposes
# (breach_percentage + mean_staff_pressure + HIGH-risk share) — so
# score_provenance = "dss_derived", not attributed to the twin.
# ---------------------------------------------------------------------------
def adapt_healthcare_evidence(evidence_list: list, scenario: str = "BASE") -> dict:
    if not evidence_list:
        raise ValueError("Empty healthcare evidence list")

    match = [e for e in evidence_list if e["scenario"] == scenario]
    if not match:
        available = [e["scenario"] for e in evidence_list]
        raise ValueError(f"Scenario {scenario!r} not found; available: {available}")
    e = match[0]

    breach_pct = e["breach_percentage"]          # 0-100
    staff_pressure = e["staff_pressure"]           # patients per staff member
    risk_dist = e.get("risk_distribution_pct", {})
    high_share = risk_dist.get("HIGH", 0.0)        # 0-100

    # DSS-derived score: starts at 100, penalized by capacity breaches (worst),
    # time spent in HIGH operational risk, and staff pressure. Weights are a
    # first pass — document/tune them the same way the retail twin documents
    # its own business_health_score() weights.
    operational_score = (
        100.0
        - 0.6 * breach_pct
        - 0.3 * high_share
        - 5.0 * max(staff_pressure - 4.0, 0.0)   # penalize only above a mild baseline load
    )
    operational_score = max(0.0, min(100.0, operational_score))

    if breach_pct > 5 or high_share > 25:
        status = "Critical"
    elif breach_pct > 0 or high_share > 10:
        status = "Warning"
    else:
        status = "Healthy"

    risk_flags = []
    if breach_pct > 0:
        risk_flags.append(f"Capacity breached {breach_pct:.1f}% of simulated hours "
                           f"(longest run {e['longest_breach']}h)")
    if high_share > 10:
        risk_flags.append(f"{high_share:.1f}% of time in HIGH operational risk")
    if staff_pressure > 4.0:
        risk_flags.append(f"Staff pressure elevated ({staff_pressure:.2f} patients/staff)")

    recommendation = (
        "Escalate: review surge staffing and diversion protocols"
        if status == "Critical"
        else "Monitor: no immediate capacity action required"
        if status == "Healthy"
        else "Watch closely: consider proactive staffing adjustment"
    )

    return _standard(
        domain="healthcare",
        entity=e.get("twin_entity", "unknown_hospital"),
        status=status,
        operational_score=operational_score,
        score_provenance="dss_derived",
        risk_flags=risk_flags,
        recommendation=recommendation,
        key_metrics={
            "scenario": e["scenario"],
            "peak_occupancy": e["peak_occupancy"],
            "mean_occupancy": e["mean_occupancy"],
            "breach_percentage": breach_pct,
            "staff_pressure": staff_pressure,
            "forecast_mae": e.get("forecast_mae"),
        },
        provenance={
            # which fixed preset (LOW/BASE/HIGH) this record is for — ASSUMED,
            # per the healthcare twin's own README: a starting condition the
            # author chose, not an observed occupancy
            "scenario": "ASSUMED",
            # occupancy trajectory + everything derived from it: produced by
            # running digital_twin.run_simulation() forward, doesn't exist in
            # source data
            "peak_occupancy": "SIMULATED",
            "mean_occupancy": "SIMULATED",
            "breach_percentage": "SIMULATED",
            "staff_pressure": "SIMULATED",
            # forecasting.py's LinearRegression MAE on held-out data
            "forecast_mae": "PREDICTED",
        },
        raw_evidence=e,
    )


# ---------------------------------------------------------------------------
# Retail / business — from retail_digital_twin/simulation.py:
# export_dss_evidence() payload + business_health_score()
#
# Native payload already carries "domain": "business" and a full impact
# comparison; business_health_score() is a separate function returning
# {"score": float 0-100, "components": {...}}. This adapter fuses the two
# (score_provenance = "twin", since business_health_score is the twin's own
# documented formula, not something invented by the DSS layer).
# ---------------------------------------------------------------------------
def adapt_retail_evidence(dss_payload: dict, health_score: dict) -> dict:
    if not dss_payload:
        raise ValueError("Empty retail DSS payload")

    score = health_score.get("score")
    if score is None:
        raise ValueError("health_score['score'] is None — was business_health_score() run "
                          "on an empty aggregate?")

    if score >= 80:
        status = "Healthy"
    elif score >= 50:
        status = "Warning"
    else:
        status = "Critical"

    scenario_agg = dss_payload.get("scenario", {})
    impact = dss_payload.get("impact", {})
    critical_entities = dss_payload.get("critical_entities", [])

    risk_flags = []
    n_critical = len(critical_entities)
    if n_critical:
        risk_flags.append(f"{n_critical} store-item pairs flagged as elevated risk")
    service_level = scenario_agg.get("service_level")
    if service_level is not None and service_level < 0.9:
        risk_flags.append(f"Service level at {service_level:.1%} under scenario "
                           f"'{scenario_agg.get('name')}'")

    recommendations = dss_payload.get("recommendations", [])
    if recommendations:
        top_rec = recommendations[0]
        # recommendations.py returns structured dicts ({finding, cause,
        # recommendation, expected_effect}), unlike the manufacturing/
        # healthcare twins' plain-string recommendations. Normalize to a
        # string here for schema consistency across domains -- the full
        # structured object is still available in raw_evidence.
        recommendation = (
            top_rec.get("recommendation", str(top_rec))
            if isinstance(top_rec, dict) else str(top_rec)
        )
    else:
        recommendation = "No specific action flagged"

    return _standard(
        domain="business",
        entity="retail_network",
        status=status,
        operational_score=score,
        score_provenance="twin",
        risk_flags=risk_flags,
        recommendation=recommendation,
        key_metrics={
            "scenario": scenario_agg.get("name"),
            "service_level": service_level,
            "stockout_events": scenario_agg.get("stockout_events"),
            "unmet_demand": scenario_agg.get("unmet_demand"),
            "score_components": health_score.get("components", {}),
        },
        provenance={
            # which named scenario preset (Baseline/Demand Surge/...) — ASSUMED,
            # a disruption the author configured, not observed
            "scenario": "ASSUMED",
            # service_level/stockouts/unmet_demand all come out of
            # DigitalTwin.run() state transitions built on top of the trained
            # XGBoost demand forecast — per the retail twin's own README table
            # ("Simulated" layer), these do not exist in the source dataset
            "service_level": "SIMULATED",
            "stockout_events": "SIMULATED",
            "unmet_demand": "SIMULATED",
            # business_health_score()'s own documented formula, applied to the
            # simulated aggregates above
            "score_components": "CALIBRATED",
        },
        raw_evidence={"dss_payload": dss_payload, "health_score": health_score},
    )
