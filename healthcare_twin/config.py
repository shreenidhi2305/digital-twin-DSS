"""
config.py
=========
Central configuration for the Healthcare Digital Twin (Streamlit edition).

Every constant below is either:
  OBSERVED    -> read directly from a dataset, no assumption involved
  ASSUMED     -> a modelling choice made because the data does not specify it
  CALIBRATED  -> derived from data + an assumption, used to scale the twin

This project is a research / FYP prototype. It does NOT make clinical
claims, does NOT diagnose, and does NOT claim real-time hospital monitoring.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# File paths (OBSERVED data sources) — relative to this file, so the app
# runs the same regardless of the working directory it's launched from.
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
PMC_DATA_PATH = DATA_DIR / "PMC_Hospital_Infrastructure.csv"
DATASET3_PATH = DATA_DIR / "dataset3.csv"

# ---------------------------------------------------------------------------
# Facility selection (fixed by original project brief; the Streamlit app
# additionally lets the user pick among ALL selectable public hospitals)
# ---------------------------------------------------------------------------
DEFAULT_FACILITY = "Kamla Nehru Hospital"

# A facility is selectable only if it is Public, has total_beds > 0, and has
# no missing values in the required staffing/resource fields. This excludes
# the private facilities in the PMC dataset that have systematically missing
# staffing data.
REQUIRED_FIELDS = [
    "Number of Beds in facility type",
    "Number of Doctors / Physicians",
    "Number of Nurses",
    "Number of Midwives Professional",
    "Average Monthly Patient Footfall",
    "Count of Ambulance",
]

# ---------------------------------------------------------------------------
# Time reference (OBSERVED / standard convention)
# ---------------------------------------------------------------------------
HOURS_PER_MONTH = 730  # standard average (365.25 days / 12 * 24)

# ---------------------------------------------------------------------------
# Demand-intensity model (CALIBRATED / ASSUMED)
# ---------------------------------------------------------------------------
# Which dataset3 variable represents temporal operational demand.
# Justification: admissions and discharges are almost collinear (r = 0.966),
# staff_count is uncorrelated noise (r ~ -0.01 to -0.03 with everything),
# flu_cases correlates moderately with admissions (r = 0.55) but is already
# substantially reflected in the admissions signal itself. Admissions is
# therefore used as the primary temporal demand driver; flu_cases is used
# separately as a forecasting feature, not as a second demand multiplier
# (to avoid double-counting the same underlying surge).
DEMAND_DRIVER_COLUMN = "admissions"

# --- Occupancy dynamics -----------------------------------------------
# occupied_beds(t+1) = occupied_beds(t) + inflow(t) - outflow(t)
#   inflow(t)  = hospital_baseline_admission_rate * demand_intensity(t)
#   outflow(t) = occupied_beds(t) / SYSTEM_ADJUSTMENT_TIME_CONSTANT_HOURS
#
# SYSTEM_ADJUSTMENT_TIME_CONSTANT_HOURS is NOT a clinical length-of-stay.
# It is a systemic parameter representing how quickly the hospital's total
# occupied-bed count adjusts toward its demand-implied level, aggregating
# discharge throughput, transfers, and elective deferral under pressure.
# ASSUMED. Exposed as an adjustable slider in the Streamlit app.
SYSTEM_ADJUSTMENT_TIME_CONSTANT_HOURS = 2500

# Target long-run occupancy rate implied by the MEAN external demand
# (demand_intensity == 1.0).
#
# ASSUMED — a literature-informed MODELLING BASELINE, not an occupancy rate
# observed at any PMC facility and NOT presented as a universally optimal
# operating point for hospitals in general. 80% is a commonly cited
# capacity-planning reference in the hospital-operations literature (headroom
# is kept for surge absorption without routinely running at/above 100%
# physical bed capacity). A different facility, case-mix, or health system
# could reasonably justify a different value -- this is a stated modelling
# choice, not a claim about the "right" occupancy target. 100% remains the
# hard physical bed-capacity limit, not a target. Kept fixed in the main UI;
# adjustable only under Advanced / Research Settings for sensitivity
# analysis, so normal users cannot silently change the core calibration.
TARGET_STEADY_STATE_OCCUPANCY = 0.80

# ---------------------------------------------------------------------------
# Scenario engine (ASSUMED simulation starting conditions)
# ---------------------------------------------------------------------------
# Fixed presets, not free sliders in the main UI -- these are assumed
# starting conditions for scenario comparison, NOT observed hospital
# occupancies at any point in time.
SCENARIOS = {
    "LOW": 0.40,     # Low-load
    "BASE": 0.60,    # Baseline
    "HIGH": 0.80,    # High-load
}

# ---------------------------------------------------------------------------
# Operational risk scoring (ASSUMED thresholds, for illustration only)
# ---------------------------------------------------------------------------
RISK_THRESHOLDS = {
    "occupancy_high": 0.85,   # operational high-pressure / warning threshold
                               # (ASSUMED reference point, distinct from the
                               # 80% steady-state modelling baseline and from
                               # the 100% physical capacity limit)
    "occupancy_breach": 1.00,  # at/above physical capacity
    "staff_pressure_high": 6.0,  # patients per staff member, ASSUMED reference
}

# ---------------------------------------------------------------------------
# Forecasting model
# ---------------------------------------------------------------------------
FORECAST_HORIZON_HOURS = 6
ADMISSION_LAGS = [1, 2, 3, 6, 12, 24]
FLU_LAGS = [1, 6, 24]
TRAIN_FRACTION = 0.8  # time-ordered split, first 80% train / last 20% test

# ---------------------------------------------------------------------------
# Output (used only when exporting evidence/reports from the app)
# ---------------------------------------------------------------------------
OUTPUT_DIR = APP_DIR / "outputs"
