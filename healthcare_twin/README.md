# Healthcare Digital Twin — Streamlit App

A lightweight simulation-based Digital Twin prototype for hospital
operational capacity and scenario analysis, presented as a single
decision-support-style dashboard.

**Not claimed:** real-time synchronization, live hospital monitoring,
actual Kamla Nehru occupancy prediction, clinical decision-making, or
clinical diagnosis/treatment. dataset3 is an external operational stream
that supplies a temporal demand *pattern* only — it does not belong to
Kamla Nehru Hospital or any other specific PMC facility. 80% steady-state
occupancy is a literature-informed modelling baseline, not an observed
rate and not claimed to be universally optimal for any hospital.

## Run it

```bash
cd twin_app
pip install -r requirements.txt
streamlit run app.py
```

Opens at `http://localhost:8501`.

## What's in here

- **`app.py`** — the Streamlit UI. Hospital selector, fixed scenario
  presets (Compare All / Low-load / Baseline / High-load), occupancy
  simulation, demand/calibration view, forecasting, and a DSS evidence
  export tab with a JSON download button. Calibration parameters
  (`TARGET_STEADY_STATE_OCCUPANCY`, `SYSTEM_ADJUSTMENT_TIME_CONSTANT_HOURS`)
  live under a collapsed **Advanced / Research Settings** sidebar
  expander — not in the main flow — so normal use can't silently change
  the twin's core assumptions.
- **Backend modules — modelling logic unchanged, docstrings sharpened
  for accurate OBSERVED/ASSUMED/CALIBRATED/SIMULATED/PREDICTED labelling:**
  `config.py`, `data_loader.py`, `hospital_profile.py`, `hospital_selector.py`,
  `demand_intensity.py`, `digital_twin.py`, `metrics.py`, `forecasting.py`,
  `dss_interface.py`.
- **`data/`** — the two source CSVs.

## Data semantics (maintained throughout app + code)

| Label | Covers |
|---|---|
| **OBSERVED** | PMC beds, doctors, nurses, midwives, ambulances, monthly footfall; dataset3 raw variables |
| **ASSUMED** | 40/60/80% scenario starting occupancies; target steady-state occupancy (0.80); system adjustment time constant (2500h) |
| **CALIBRATED / DERIVED** | Baseline demand, demand intensity, admission-rate calibration |
| **SIMULATED** | Occupied beds, occupancy rate, available beds, capacity breaches, staff pressure, operational risk |
| **PREDICTED** | 6-hour-ahead admissions/demand forecast |

Monthly patient footfall is never described as inpatient admissions — it's
an observed facility-activity/plausibility metric only; baseline
simulation demand is derived from the occupancy model's own steady-state
relationship (`target_occupancy × bed_capacity = baseline_demand ×
adjustment_timescale`), not from footfall.

## Occupancy model

Kept as-is: an **occupancy-dependent first-order outflow/turnover model**
(leaky-bucket state update) — `occupied_beds(t+1) = occupied_beds(t) +
simulated_demand(t) - occupied_beds(t)/time_constant`. Not a literal
replay of dataset3's admissions−discharges (non-negative in all 10,000
raw rows, which would make occupancy grow unbounded), and not described
as a Little's Law result — no queueing-theory equivalence is derived or
claimed, "first-order outflow/turnover" is the accurate name.

## Forecasting

Unchanged LinearRegression, time-ordered 80/20 validation, actual
current MAE/R² reported without embellishment. Described as a
supplementary future-demand signal for the DSS — not a clinical
prediction system, and not the direct driver of the occupancy simulator
(that's `demand_intensity`, derived from dataset3's temporal pattern).

## DSS interface

Unchanged: one structured evidence record per scenario (peak/mean/min
occupancy, breach hours/percentage, longest breach, max capacity excess,
staff pressure, forecast performance). No composite "readiness score" is
invented — the app surfaces exactly what `dss_interface.build_evidence`
defines.

## UI fixes (this revision)

- **Light theme forced app-wide** via `.streamlit/config.toml` — the white-on-white
  text was Streamlit auto-applying the visitor's OS/browser dark-mode theme to
  chart legends/axes while our own CSS assumed light, producing near-invisible
  light-gray-on-white text. The app no longer depends on the viewer's system theme.
- Every chart now renders with `theme=None` passed to `st.plotly_chart` (so
  Streamlit doesn't re-theme on top of our explicit colors) and explicit
  font/legend/axis/hover colors set directly on the figure, so this can't
  silently regress again if the theme changes.

## Time-axis navigation

The occupancy, demand-intensity, and forecast charts now carry a range
selector (24h / 7d / 30d / 3m / All buttons) and a range slider under the
chart, with tick labels that adapt automatically — hours when zoomed into a
day, dates across weeks, months when fully zoomed out. Data supports this
down to the hour: dataset3 is a clean, gapless hourly series (2019-01-01 to
2020-02-21, no missing timestamps), so any zoom level is real data, not
interpolation. The forecast tab opens zoomed to the last 14 days of the
held-out test period by default (zoom out with "All" for the full ~83 days).

## What changed in this revision

1. `TARGET_STEADY_STATE_OCCUPANCY` default: 1.00 → **0.80** (literature
   baseline, not optimum); `RISK_THRESHOLDS["occupancy_high"]`: 0.90 → **0.85**
   (operational high-pressure warning line, distinct from the 80%
   modelling baseline and the 100% physical limit).
2. Calibration sliders (target occupancy, time constant) moved out of the
   main UI into a collapsed **Advanced / Research Settings** expander.
3. LOW/BASE/HIGH sliders replaced with **fixed presets** (40/60/80%) plus
   a scenario radio: pick one, or **Compare All Scenarios**.
4. Occupancy model renamed/documented as "occupancy-dependent first-order
   outflow/turnover model"; Little's Law framing removed.
5. Baseline demand docstrings rewritten to make the CALIBRATED/DERIVED
   steady-state formula explicit and stop implying footfall is admissions.
6. Forecast description clarified: supplementary signal, not the
   occupancy simulator's driver.
7. UI restyled white/blue (was dark), with an Observed / Assumed /
   Calibrated-Derived / Simulated legend row up top and a warning-level
   line drawn on the occupancy chart at 85%.

Nothing about the underlying architecture changed: same datasets, same
demand-intensity derivation, same occupancy dynamics, same
LinearRegression forecaster, same DSS evidence schema.

## Data connector (this revision)

Added `data_connector.py` — an explicit `DataConnector` interface with a
concrete `CSVReplayConnector` (what this app actually runs on) and a
documented `LiveFeedConnector` stub that lists exactly what a real
hospital-feed integration would require (HL7 FHIR / ADT, auth, polling,
message ordering) and raises `NotImplementedError` rather than faking it.
`data_loader.py` is now a thin wrapper over the default connector, kept
for backward compatibility. The active connector and its `source_type`
are shown in the sidebar, so the batch-vs-live distinction is visible in
the running app, not just asserted in the report.

## Sensitivity analysis (this revision)

Demand & Calibration tab now includes a "Sensitivity analysis" panel that
re-simulates the Baseline scenario across a grid of system adjustment
time constants (500h–5000h), holding target occupancy fixed. Verified
finding: because baseline demand is itself derived as `target_occupancy ×
beds / time_constant`, inflow and outflow both scale with `1/time_constant`
— the model is bounded and mean-reverting across the whole tested range
(no runaway growth at high time constants). What actually changes with
time constant is relaxation speed: short time constants let the assumed
starting occupancy decay toward the target within days (closing the
LOW/BASE/HIGH gap fast); long time constants make the system move slowly
relative to the ~14-month dataset horizon, so the starting-occupancy
assumption persists longer and peak occupancy is damped further, not
amplified. No capacity breach occurs anywhere in the tested range for
Kamla Nehru Hospital under dataset3's demand-intensity fluctuations
(spikes up to ~1.8× the mean, but rarely sustained beyond ~5 hours) —
this is a genuine finding about the current assumptions' headroom, not a
bug.
