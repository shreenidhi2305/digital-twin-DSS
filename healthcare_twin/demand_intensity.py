"""
demand_intensity.py
=====================
Converts the external dataset3 operational stream into a normalized,
dimensionless temporal demand pattern, then translates that pattern into
an hospital-specific simulated admission rate for the selected facility.

Variable selection rationale (see report for full correlation table):
  - admissions vs discharges   r = 0.966  -> near-redundant, admissions used
  - admissions vs staff_count  r = -0.015 -> staff_count is uncorrelated
                                              noise, not a demand signal
  - admissions vs flu_cases    r = 0.548  -> flu_cases is a moderate
                                              co-driver but its signal is
                                              already substantially present
                                              in admissions; using both as
                                              independent demand multipliers
                                              would double-count the same
                                              surge, so flu_cases is instead
                                              reserved as a forecasting
                                              feature only.

Therefore DEMAND_DRIVER_COLUMN = "admissions" is used as the sole basis for
DemandIntensity(t).

    DemandIntensity(t) = admissions(t) / mean(admissions)

This is dimensionless and centered at 1.0 by construction, which is what
lets it be transplanted onto a completely different hospital's scale
without inheriting dataset3's absolute admission counts. dataset3 is an
external operational stream supplying a temporal PATTERN only -- these
admission counts do NOT belong to Kamla Nehru Hospital or any other
specific PMC facility, and are never described as such.

CALIBRATED / DERIVED baseline demand -- NOT an observed quantity.
The hospital-specific baseline inpatient demand (patients/hour) is
derived from the steady-state relationship implied by the occupancy
model itself, not from monthly footfall:

    target_occupancy * bed_capacity = baseline_demand * adjustment_timescale
    =>  baseline_demand = target_occupancy * bed_capacity / adjustment_timescale

    simulated_demand(t) = baseline_demand * DemandIntensity(t)

target_occupancy (TARGET_STEADY_STATE_OCCUPANCY) and adjustment_timescale
(SYSTEM_ADJUSTMENT_TIME_CONSTANT_HOURS) are both explicit, documented
ASSUMED/CALIBRATED parameters -- see config.py. The PMC "Average Monthly
Patient Footfall" field is NOT used to derive baseline_demand and is NOT
described as inpatient admissions; it mixes outpatient and inpatient
activity that this dataset does not disaggregate. Footfall is used only
in `implied_admission_fraction_of_footfall` below, as an independent
plausibility check on the calibration -- never as a second free
parameter the calibration is fit to.
"""

import pandas as pd
import config


def compute_demand_intensity(ops_df: pd.DataFrame) -> pd.DataFrame:
    """Add a dimensionless demand_intensity column to the operational stream."""
    df = ops_df.copy()
    driver = df[config.DEMAND_DRIVER_COLUMN]
    baseline = driver.mean()
    df["demand_baseline"] = baseline
    df["demand_intensity"] = driver / baseline
    return df


def hospital_baseline_admission_rate(total_beds: int,
                                      target_steady_state_occupancy: float = config.TARGET_STEADY_STATE_OCCUPANCY,
                                      time_constant_hours: float = config.SYSTEM_ADJUSTMENT_TIME_CONSTANT_HOURS
                                      ) -> float:
    """
    CALIBRATED / DERIVED -- not observed. Hospital-specific baseline
    inpatient demand, in patients/hour, derived from the steady-state
    relationship so that under mean external demand (demand_intensity ==
    1.0) the occupancy dynamics settle at target_steady_state_occupancy:

        target_occupancy * bed_capacity = baseline_demand * adjustment_timescale
        baseline_demand = target_occupancy * total_beds / time_constant_hours

    target_steady_state_occupancy is a literature-informed modelling
    baseline (default 0.80, see config.py) -- not an observed occupancy
    rate and not claimed to be universally optimal.
    """
    return target_steady_state_occupancy * total_beds / time_constant_hours


def implied_admission_fraction_of_footfall(rate: float, monthly_footfall: float) -> float:
    """
    Plausibility check ONLY: what fraction of the hospital's OBSERVED
    monthly patient footfall the CALIBRATED baseline demand above would
    imply, if that footfall were entirely inpatient admissions. Not used
    to derive baseline_demand, and monthly footfall is never itself
    described as an inpatient-admission count -- it is a general
    facility-activity metric (outpatient + inpatient, undisaggregated in
    this dataset). Used only to sanity-check that the calibration is
    plausible for a general hospital's OPD:IPD mix.
    """
    monthly_rate_equivalent = rate * config.HOURS_PER_MONTH
    return monthly_rate_equivalent / monthly_footfall


def compute_simulated_demand(ops_df: pd.DataFrame, total_beds: int, monthly_footfall: float,
                              target_steady_state_occupancy: float = config.TARGET_STEADY_STATE_OCCUPANCY,
                              time_constant_hours: float = config.SYSTEM_ADJUSTMENT_TIME_CONSTANT_HOURS
                              ) -> pd.DataFrame:
    """
    Full pipeline: demand_intensity(t) -> hospital-specific simulated_demand(t)
    (patients/hour who plausibly require a bed, for the selected hospital).
    """
    df = compute_demand_intensity(ops_df)
    rate = hospital_baseline_admission_rate(total_beds, target_steady_state_occupancy, time_constant_hours)
    df["hospital_baseline_admission_rate"] = rate
    df["simulated_demand"] = rate * df["demand_intensity"]
    return df
