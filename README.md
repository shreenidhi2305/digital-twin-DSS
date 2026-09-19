# AI-Based Intelligent Decision Support System — Unified DSS Project

This folder combines your three Digital Twin prototypes (manufacturing,
healthcare, business/retail) and your NLP pipeline's output into one fused
Digital Twin Readiness Index, shown on one Streamlit dashboard, and lets you
open each twin's own full dashboard from that same app.

## Folder layout — everything is already in the right place

```
dss_project/
├── dashboard.py          <- run THIS: entry point + navigation between pages
├── dss_overview.py        <- "DSS Overview" page (Step 4: the fused Readiness Index)
├── twin_embed.py          <- registry (TWINS) + runner that opens each twin's own dashboard
├── .streamlit/config.toml <- light theme for the whole app
├── dss_engine.py          <- Step 2: NLP + twin fusion logic
├── evidence_schema.py     <- Step 1: common evidence schema + adapters
├── twin_runners.py        <- Step 3: real (non-mocked) twin integration
├── master_nlp_aspects.csv <- your NLP pipeline's aspect-sentiment output
├── requirements.txt
├── manufacturing_twin/    <- trimmed copy of manufacturing_twin (no venv/)
│   ├── predictor.py, twin.py, stream.py, dashboard_standalone.py, ...
│   ├── models/            <- the actual trained XGBoost model + encoders
│   └── data/live_machine_stream.csv
├── healthcare_twin/        <- full copy of twin_app
│   ├── app.py, config.py, digital_twin.py, dss_interface.py, ...
│   └── data/ (PMC_Hospital_Infrastructure.csv, dataset3.csv)
└── retail_twin/             <- full copy of retail_digital_twin
    ├── app.py, twin.py, simulation.py, recommendations.py, ...
    ├── models/demand_model.pkl
    └── data/ (train.csv, test.csv)
```

**Nothing needs to be moved.** `dss_overview.py` looks for `manufacturing_twin/`,
`healthcare_twin/`, `retail_twin/` and `master_nlp_aspects.csv` right next to
itself (see `TWIN_DIRS` / `ASPECTS_CSV` at the top of `dss_overview.py`) — that's
exactly this layout. If you ever move a twin folder elsewhere, update the
matching path in `TWIN_DIRS` instead of moving files back.

One thing I *did* trim: the manufacturing twin's original zip included a full
Python `venv/` (~245 MB) bundled inside it. That's your local virtual
environment, not project code — it doesn't travel with the project and isn't
included here. Everything the twin actually needs to run (`predictor.py`,
`twin.py`, `stream.py`, the trained model files, the sensor CSV) is included.

## 1. Install dependencies

From inside `dss_project/`:

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

This installs pandas, numpy, scikit-learn, xgboost, joblib, streamlit and
plotly (the twin dashboards need plotly).

## 2. Run the unified dashboard

```bash
streamlit run dashboard.py
```

Opens at `http://localhost:8501`. The top navigation bar has one page per
dashboard:

| Page | What it is |
|---|---|
| **DSS Overview** | The fused NLP + twin Readiness Index (below). Also has "Open ..." buttons for each twin. |
| **Healthcare Twin** | The healthcare twin's own full dashboard (`healthcare_twin/app.py`) |
| **Retail Twin** | The retail twin's own full dashboard (`retail_twin/app.py`) |

The twin pages run each twin's original `app.py` unchanged, so they behave
exactly as they do standalone. See `twin_embed.py` for how they are isolated
from each other (working directory, module names, page config).

On the DSS Overview page, first load will:
1. Run the manufacturing twin against `data/live_machine_stream.csv` using
   the real trained XGBoost model
2. Run the healthcare twin's simulation for the default facility (Kamla
   Nehru Hospital) across the LOW/BASE/HIGH presets
3. Run the retail twin's simulation for a sample network (3 stores × 10
   items) using the real trained demand model
4. Load `master_nlp_aspects.csv` and aggregate sentiment per sector
5. Fuse each sector's NLP + twin evidence into a Readiness Index

This first run takes a little longer (loading three models, running three
simulations) — results are cached for the rest of the session. Use the
"🔄 Re-run all twins" button in the app to force a fresh run.

## 3. If something is missing

`twin_runners.py`'s three `run_*_twin()` functions check for their required
files up front and raise a clear `FileNotFoundError` naming exactly what's
missing, rather than failing partway through — so if a folder got moved or a
file didn't copy, the dashboard's error banner will tell you precisely which
file to restore and where.

## Adding the manufacturing twin's dashboard

Once its dashboard is final, put the files in `manufacturing_twin/`, then in
`twin_embed.py` uncomment the `"manufacturing"` entry in `TWINS` (check that
`entry` names the right script, currently `dashboard_standalone.py`). The
navigation bar and the buttons on the overview page pick it up automatically.
No other file needs to change.

## What you can safely add or swap later

- **A different hospital**: `run_healthcare_twin(twin_dir, hospital_name="...")`
  in `twin_runners.py` — any facility name from
  `healthcare_twin/data/PMC_Hospital_Infrastructure.csv`.
- **A different retail scenario**: `run_retail_twin(twin_dir, scenario_name=...,
  demand_pct=..., lead_time_delta=...)` — same presets as `retail_twin/app.py`'s
  sidebar (Demand Surge, Supply Disruption, Combined Disruption, or Custom).
- **A different machine / longer replay**: `run_manufacturing_twin(twin_dir,
  machine_id=..., n_readings=...)`.
- **Updated NLP results**: just overwrite `master_nlp_aspects.csv` with a
  fresh run of your `aspect_sentiment.py` pipeline — `dss_engine.py` re-reads
  it fresh on every dashboard run (nothing is cached across files).

## Running a single twin's own standalone app (optional)

Each twin still has its own original Streamlit app if you want to demo it
individually rather than through the unified dashboard:

```bash
cd manufacturing_twin && streamlit run dashboard_standalone.py
cd healthcare_twin    && streamlit run app.py
cd retail_twin        && streamlit run app.py
```

The unified app's overview page calls their underlying modules directly, and
the twin pages run their `app.py`, so both can be used independently without
conflict. Two small differences from the earlier copies: `retail_twin/app.py`
is the newer standalone version (dark-mode readability CSS, days-of-supply
metric), and `healthcare_twin/app.py` has a one-line fix in the Forecasting tab
(`ts.iloc[...]` on a `DatetimeIndex` raised an AttributeError).
