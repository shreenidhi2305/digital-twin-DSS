"""
digital_twin.py
=================
Core simulation of the virtual hospital's bed occupancy state.

Model: an occupancy-dependent first-order outflow/turnover model (a
leaky-bucket style state update), NOT a literal replay of dataset3's
admissions-minus-discharges (which is structurally unsuitable -- see
project brief: net flow is >=0 in every one of the 10,000 raw
observations, mean +2.02/hr, so directly applying it makes occupancy
grow without bound). No Little's Law equivalence is claimed or derived
here -- "occupancy-dependent first-order outflow" is the accurate
description; it is not renamed to invoke queueing theory it hasn't been
shown to satisfy.

Instead:

    inflow(t)  = simulated_demand(t)          [from demand_intensity.py,
                                                driven by dataset3's temporal
                                                pattern, scaled to this
                                                hospital via monthly footfall]

    outflow(t) = occupied_beds(t) / SYSTEM_ADJUSTMENT_TIME_CONSTANT_HOURS
                                               [occupancy-dependent
                                                first-order outflow/turnover
                                                model: outflow is
                                                proportional to current
                                                occupancy. This is NOT
                                                presented as a Little's Law
                                                result -- no queueing-theory
                                                equivalence is derived or
                                                claimed here, it is simply a
                                                first-order relaxation term.
                                                NOT a clinical length-of-stay
                                                either: a systemic,
                                                ASSUMED/CALIBRATED time
                                                constant, see config.py for
                                                rationale]

    occupied_beds(t+1) = occupied_beds(t) + inflow(t) - outflow(t)

Because outflow is proportional to current occupancy, the system is
self-correcting (a rising occupied-bed count increases its own outflow),
which is what keeps it bounded without ever clipping occupancy to
capacity. Breaches above capacity and dips below zero are recorded, not
hidden.
"""

import numpy as np
import pandas as pd
import config


def run_simulation(ops_df: pd.DataFrame, total_beds: int, doctors: int,
                    nurses: int, midwives: int, initial_occupancy_pct: float,
                    time_constant_hours: float = config.SYSTEM_ADJUSTMENT_TIME_CONSTANT_HOURS
                    ) -> pd.DataFrame:
    """
    Run the bed-occupancy twin over the full external time series for one
    initial-occupancy scenario. Requires ops_df to already contain a
    'simulated_demand' column (patients/hour), see demand_intensity.py.

    Returns a DataFrame with the full simulated Digital Twin state at every
    timestep. occupied_beds is NOT clipped at total_beds (breaches are kept
    real); it IS floored at 0 (a hospital cannot hold negative patients),
    with each such event separately logged as a floor_event.
    """
    n = len(ops_df)
    demand = ops_df["simulated_demand"].to_numpy()

    occupied = np.empty(n)
    floor_event = np.zeros(n, dtype=bool)
    raw_before_floor = np.empty(n)

    total_staff = doctors + nurses + midwives
    occ0 = initial_occupancy_pct * total_beds
    prev = occ0

    for t in range(n):
        inflow = demand[t]
        outflow = prev / time_constant_hours
        nxt = prev + inflow - outflow
        raw_before_floor[t] = nxt
        if nxt < 0:
            floor_event[t] = True
            nxt = 0.0
        occupied[t] = nxt
        prev = nxt

    out = ops_df.copy()
    out["hospital_name"] = None  # filled by caller
    out["total_beds"] = total_beds
    out["doctors"] = doctors
    out["nurses"] = nurses
    out["midwives"] = midwives
    out["initial_occupancy_pct"] = initial_occupancy_pct
    out["occupied_beds"] = occupied
    out["occupied_beds_raw"] = raw_before_floor  # before floor clipping
    out["occupancy_rate"] = out["occupied_beds"] / total_beds
    out["available_beds"] = total_beds - out["occupied_beds"]
    out["capacity_breach"] = out["occupied_beds"] > total_beds
    out["capacity_excess"] = np.where(out["capacity_breach"],
                                       out["occupied_beds"] - total_beds, 0.0)
    out["floor_event"] = floor_event
    out["staff_pressure"] = out["occupied_beds"] / total_staff  # patients per staff

    high_occ = out["occupancy_rate"] >= config.RISK_THRESHOLDS["occupancy_high"]
    breach = out["capacity_breach"]
    high_staff = out["staff_pressure"] >= config.RISK_THRESHOLDS["staff_pressure_high"]

    risk = pd.Series("LOW", index=out.index)
    risk[high_occ | high_staff] = "MEDIUM"
    risk[breach] = "HIGH"
    out["operational_risk"] = risk

    return out
