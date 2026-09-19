"""
dss_interface.py
==================
Assembles the structured evidence objects the Healthcare Digital Twin
hands off to the larger Decision Support System (DSS). Pure formatting /
aggregation layer -- no new computation happens here.
"""

import config


def build_evidence(profile, summaries: dict, forecast_result: dict) -> list:
    """One structured evidence record per scenario, per the DSS schema."""
    evidence = []
    for name, s in summaries.items():
        evidence.append({
            "domain": "healthcare",
            "twin_entity": profile.name,
            "scenario": name,
            "initial_occupancy": s["initial_occupancy"],
            "peak_occupancy": round(s["peak_occupancy"], 4),
            "mean_occupancy": round(s["mean_occupancy"], 4),
            "min_occupancy": round(s["min_occupancy"], 4),
            "capacity_breach_count": s["capacity_breach_count"],
            "total_breach_hours": s["total_breach_hours"],
            "breach_percentage": round(s["breach_percentage"], 2),
            "longest_breach": s["longest_breach_hours"],
            "maximum_capacity_excess": round(s["maximum_capacity_excess"], 2),
            "floor_event_count": s["floor_event_count"],
            "staff_pressure": round(s["mean_staff_pressure"], 3),
            "peak_staff_pressure": round(s["peak_staff_pressure"], 3),
            "risk_distribution_pct": s["risk_distribution"],
            "forecast_mae": round(forecast_result["admissions_plus_flu"]["mae"], 3),
            "forecast_r2": round(forecast_result["admissions_plus_flu"]["r2"], 3),
            "forecast_description": forecast_result["description"],
            "notes": (
                "occupied_beds/occupancy_rate are SIMULATED, not observed. "
                "initial_occupancy is an ASSUMED scenario starting condition. "
                "demand pattern is derived from an external operational "
                "dataset (dataset3), not from this hospital's own history."
            ),
        })
    return evidence
