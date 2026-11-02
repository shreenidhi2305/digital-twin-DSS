# Manufacturing Digital Twin

Anomaly detection on the MetroPT-3 metro-train air compressor (APU), used as
an evidence source for the AI-based decision support system.

This is an anomaly alarm from a frozen sparse autoencoder. It is not a
failure prediction and not a remaining-useful-life estimate.

## Data and model

The model was trained offline on MetroPT-3. Training code is not part of
this runtime. Artifacts live in `models/metropt3_sae/` and are checked
against `MANIFEST.sha256` every time the detector loads. A mismatch stops
scoring.

| Piece | Role |
|---|---|
| `sae_model_state.pt` | `Linear(23, 64) -> ReLU -> Linear(64, 23)` |
| `scaler.json` | Frozen z-score. Never refit. |
| `feature_names.json` | 23 inputs, fixed order |
| `thresholds.json` | Frozen thresholds for 0.5 / 1.0 / 2.0 episodes per day |
| `run_config.json` | Architecture, 30 min episode merge, 30 s segment gap |

The anomaly score is the mean squared reconstruction error of the 23
standardised features. The primary threshold is budget **1.0 episode/day**
(about 0.02375). Budgets 0.5 and 2.0 are selectable. Alarms within 30
minutes are one episode.

The reference replay file is **not** in git. Download MetroPT-3 from the
UCI repository (dataset 791) and save the CSV as one of:

- `data/MetroPT3_AirCompressor_.csv`
- `data/MetroPT3(AirCompressor).csv`

The published file is already one row every 10 seconds. The feature
pipeline still accepts a faster log and averages it into 10 s bins.

### Features

Eight raw signals (TP2, TP3, H1, DV_pressure, Oil_temperature,
Motor_current, COMP, DV_eletric), `tp3_minus_h1`, `d_TP3_per_s`,
`load_duty_30m`, and a 5-minute trailing mean and sample standard deviation
of TP2, TP3, H1, DV_pressure, Oil_temperature and Motor_current.

Excluded on purpose: Reservoirs, Towers, MPG, LPS, Pressure_switch,
Oil_level, Caudal_impulses.

Windows are trailing only. A gap longer than 30 seconds starts a new
segment. A row without a full 30-minute window, or with a missing sensor,
is marked invalid and is not scored. Missing values are not filled in.

Two definitions had to be taken from the dataset description and the
frozen scaler, because the training notebook was not in the artifact
bundle:

- `load_duty_30m` is the trailing 180-sample mean of `DV_eletric`.
  MetroPT-3 defines that signal as active while the compressor is under
  load, and the scaler's mean for this feature is about 0.121, the same
  place as `DV_eletric`. `COMP` is the opposite signal (active when the
  unit is off or offloaded).
- `d_TP3_per_s` is the change in TP3 divided by the elapsed seconds since
  the previous 10 s bin, inside the current segment.
- The 5-minute statistics use 30 trailing bins. Standard deviation uses
  the sample divisor (ddof=1).
- Operating state is a display label, not a model input. Current below
  2 A is `off`. Otherwise `COMP < 0.5` is `load`, and the rest is
  `offload`.

`features.parquet` was not in the bundle, so these definitions are not
bit-matched to the training table. See the test report for the parity
status.

## What the published test actually did

Test period 2020-05-29 to 2020-09-01, reported events E2, E3 and E4.
E1 is in validation. Reported times are not verified onsets. The timezone
is unknown, E2's maintenance note ("30 Apr") does not match its reported
interval, and E1 is day-level only.

| Budget (episodes/day) | Test result |
|---|---|
| 0.5 | Detects E3 only, weakly. About 0.24 false-alarm episodes/day. |
| 1.0 (primary) | Detects E3 only. First alarm about 4 s after the reported start, about 20% of rows in that reported interval. Misses E2 and E4. About 0.27 false-alarm episodes/day. |
| 2.0 | Also catches E2, still misses E4, at about 1.9 false-alarm episodes/day. |

Episode-overlap precision (about 0.48 / 0.17 / 0.03) is a lower bound,
because periods treated as normal were not verified.

The budget 1.0 threshold is sensitive: validation alarm rate about 0.22
episodes/day in the first half and about 1.61 in the second. Monthly p95
scores fall from about 1e-2 (April–May) to about 1e-4 (July–August).
From train to validation, DV_pressure's standard deviation grows by about
2.1 times, and about 4% of TP3 values leave the training range.

There are four reported events, all air leaks. That supports an
anomaly-detection demo on real data. It does not support a claim of
reliable fault prediction.

The dashboard shows a "model may be drifting" banner when the alarm rate
over a revealed span of at least 7 days is more than twice the selected
budget.

## Removed from the previous twin

The previous module scored a synthetic AI4I-style stream with XGBoost.
These outputs are gone, and nothing was invented to stand in for them:

- failure probability and predicted failure class
- tool wear
- the 0.30 / 0.60 health bands
- the 0–100 score defined as `100 * (1 - failure probability)`
- air temperature, process temperature, rotational speed and torque

The DSS readiness index for manufacturing uses the literature/news
component only. The anomaly alarm is reported beside it and is not folded
into that 0–100 number.

## Architecture

```
data/MetroPT3_AirCompressor_.csv     reference replay (you download this)
models/metropt3_sae/                 frozen weights, scaler, thresholds, manifest
dt/features/metropt3.py              trailing features, batch and stream
dt/models/metropt3_sae.py            SAE score, contributions, threshold
dt/episodes.py                       30 min episode merge, drift rate
twin.py                              live state and DSS packet
stream.py                            chronological CSV replay
dashboard_standalone.py              Streamlit replay
```

## Running it

From the repository root, with a virtual environment:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cd manufacturing_twin
python check_environment.py
streamlit run dashboard_standalone.py
```

`check_environment.py` confirms the checksums and that a forward pass
runs on CPU. The dashboard needs the CSV as well.

Playback speed is the pause between steps, and each step can reveal
several 10 s bins. The primary budget is selected by default.

## DSS packet

`twin.evidence_for_dss()` returns the current bin only, for example:

```json
{
  "domain": "Manufacturing",
  "machine": "APU-01",
  "alarm": false,
  "anomaly_score": 0.000012,
  "threshold": 0.02375,
  "budget": "1.0",
  "operating_state": "offload",
  "top_features": ["Motor_current", "TP3_std_5m", "d_TP3_per_s"],
  "active_episode": null,
  "data_quality": [],
  "model_may_be_drifting": false,
  "score_kind": "reconstruction_mse"
}
```

## Tests

From the repository root, with the manufacturing folder on `PYTHONPATH`
(the test file inserts it):

```bash
python -m pytest manufacturing_twin/tests/test_metropt3.py
```

Batch and streaming features are compared on a synthetic 10 s grid and on
a 1 Hz grid. If the MetroPT-3 CSV is present, a further test scores
2020-02-01 through 2020-02-08 and checks the score distribution against
the published monthly drift table. That is not a row-level match to the
training features. `features.parquet` was not available.

A replay of 2020-06-04 to 2020-06-06 at budget 1.0 opened one alarm episode
at 2020-06-05 09:55:07 and kept it open through the end of 5 June. The
published test's first E3 alarm is 2020-06-05 10:00:04, split into four
episodes. The top features are the same ones named on the published E3
episodes (Motor_current, TP3_std_5m, d_TP3_per_s). Episode start times
were not reproduced.
