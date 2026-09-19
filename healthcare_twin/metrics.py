"""
metrics.py
===========
Computes scenario-level summary statistics from a simulated twin
trajectory (the DataFrame produced by digital_twin.run_simulation).
"""

import numpy as np
import pandas as pd


def _longest_true_run(mask: pd.Series) -> int:
    """Length, in consecutive rows, of the longest run of True in a boolean series."""
    if not mask.any():
        return 0
    groups = (mask != mask.shift()).cumsum()
    run_lengths = mask.groupby(groups).transform("size")[mask]
    return int(run_lengths.max())


def summarize_scenario(sim_df: pd.DataFrame) -> dict:
    breach = sim_df["capacity_breach"]
    n = len(sim_df)

    return {
        "peak_occupancy": float(sim_df["occupancy_rate"].max()),
        "min_occupancy": float(sim_df["occupancy_rate"].min()),
        "mean_occupancy": float(sim_df["occupancy_rate"].mean()),
        "capacity_breach_count": int(breach.sum()),
        "total_breach_hours": int(breach.sum()),
        "breach_percentage": float(breach.mean() * 100),
        "longest_breach_hours": _longest_true_run(breach),
        "maximum_capacity_excess": float(sim_df["capacity_excess"].max()),
        "floor_event_count": int(sim_df["floor_event"].sum()),
        "mean_staff_pressure": float(sim_df["staff_pressure"].mean()),
        "peak_staff_pressure": float(sim_df["staff_pressure"].max()),
        "n_hours_simulated": n,
        "risk_distribution": sim_df["operational_risk"].value_counts(normalize=True)
                                                          .mul(100).round(2).to_dict(),
    }
