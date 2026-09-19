# Manufacturing Digital Twin

Lightweight digital twin prototype for the manufacturing domain, built as an
evidence source for the AI-Based Decision Support System (DSS).

## What this is (and isn't)

This is **not** the primary deliverable of the project — it's a small,
working demonstration that:
1. mirrors a physical machine's state in near-real-time,
2. predicts failure risk from live sensor readings, and
3. hands the DSS a compact, decision-ready evidence packet — not raw telemetry.

## Data & model

Built around the AI4I 2020 Predictive Maintenance-style schema:

| Feature | Meaning |
|---|---|
| `Type` | Product quality variant, encoded L=1 / M=2 / H=0 (see `label_encoder.pkl`) |
| `Air temperature [K]` | Ambient air temperature |
| `Process temperature [K]` | Process temperature |
| `Rotational speed [rpm]` | Spindle speed |
| `Torque [Nm]` | Applied torque |
| `Tool wear [min]` | Cumulative tool wear |

The model is a binary `XGBClassifier` predicting `Machine failure` (0/1),
stored as `machine_failure_xgboost.json` — XGBoost's own portable
serialization format, which (unlike a pickled booster) loads correctly
regardless of which xgboost version is installed locally. A `.pkl` copy is
also produced by the training notebook for archival purposes, but the app
only reads the `.json`. `predictor.py` handles renaming the bracketed CSV
column names to the plain names the model was trained on, so you can feed
it either format.

## Architecture

```
data/live_machine_stream.csv       → historical/live sensor rows
models/machine_failure_xgboost.json → trained model (portable format, used by the app)
models/machine_failure_xgboost.pkl  → archival copy only, not read by the app
models/label_encoder.pkl, feature_columns.pkl → label encoder, feature schema
predictor.py                   → loads artifacts, scores one reading
stream.py                      → replays the CSV as a simulated live feed
twin.py                        → MachineTwin: holds live state, classifies
                                  health, generates recommendation + DSS evidence
dashboard.py                   → Streamlit UI wrapping all of the above
```

## Health classification

Failure probability is bucketed into:

| Probability | Health |
|---|---|
| < 0.30 | Healthy |
| 0.30 – 0.60 | Warning |
| ≥ 0.60 | Critical |

These thresholds are placeholders defined in `twin.py`
(`HEALTHY_THRESHOLD`, `WARNING_THRESHOLD`). The training data is
imbalanced (~2% positive class), so it's worth tuning these against a
precision-recall curve on your validation set rather than trusting 0.5 as
a cutoff, and reporting that tuning process in your dissertation.

## Evidence sent to the DSS

`twin.evidence_for_dss()` returns exactly the compact packet the DSS layer
should consume, e.g.:

```json
{
  "domain": "Manufacturing",
  "machine": "CNC-01",
  "health": "Warning",
  "failure_probability": 0.42,
  "tool_wear": 187,
  "recommendation": "Schedule inspection within the next maintenance window"
}
```

## Running it

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS/Linux

pip install -r requirements.txt
python check_environment.py    # sanity-check the model loads before launching the UI
streamlit run dashboard.py
```

### Note on `XGBoostError: input stream corrupted`

Earlier versions of this project loaded the model from a pickled
`XGBClassifier` (`machine_failure_xgboost.pkl`). Pickling a booster ties the
file to the exact xgboost version that wrote it, so loading it with any
other version raised `XGBoostError: input stream corrupted`. The project
now saves and loads `machine_failure_xgboost.json` instead — XGBoost's own
portable format — which any reasonably recent xgboost version can read, so
this class of error shouldn't come up anymore.

`requirements.txt` still pins `xgboost==3.3.0` and `scikit-learn==1.6.1` to
match the Colab training environment, mainly to keep behavior consistent
rather than to avoid a loading error. Run `check_environment.py` any time
you move the project to a new machine or rebuild the venv — it isolates the
model-loading step so any file/path problem shows up immediately instead of
inside a Streamlit traceback.

Use the sidebar to Play/Pause the stream, adjust playback speed, and reset.
The dashboard shows current health, operating conditions, a failure-probability
timeline, and the live JSON evidence packet.

## Wiring into the DSS

For your DSS module, the integration point is simply:

```python
from twin import MachineTwin
import stream

twin = MachineTwin("CNC-01")
for reading in stream.replay():
    twin.update(reading)
    evidence = twin.evidence_for_dss()
    # → hand `evidence` to your DSS aggregator alongside the
    #   NLP-derived sentiment/trend signals from academic papers + news
```
