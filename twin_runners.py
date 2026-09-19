"""
twin_runners.py
=================
Step 3 of the DSS fusion layer: REAL (non-mocked) integration.

Each function here runs one twin's actual pipeline -- the same modules and
the same call order its own app.py/dashboard.py uses -- and returns the
twin's genuine evidence output. Nothing is faked or hand-written; if a twin's
data or model files are missing, these functions raise loudly rather than
silently falling back to a placeholder.

Because each twin is its own project with its own local modules of the same
names (e.g. every twin has some form of "config.py"), each run_*_twin()
function does its imports *inside* the function, after temporarily pointing
sys.path at that one twin's directory, and cleans sys.path/sys.modules up
afterwards. This keeps the three twins from clobbering each other's modules
if all three are run in the same process (as the unified dashboard does).

Usage:
    from twin_runners import run_manufacturing_twin, run_healthcare_twin, run_retail_twin

    mfg_evidence = run_manufacturing_twin(MANUFACTURING_DIR)
    hc_evidence_list = run_healthcare_twin(HEALTHCARE_DIR)
    dss_payload, health_score = run_retail_twin(RETAIL_DIR)
"""

import sys
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def _isolated_path(twin_dir: Path, module_names):
    """
    Temporarily prepend twin_dir to sys.path so its local modules import
    correctly, and remove those module names from sys.cache afterwards so
    the *next* twin's same-named modules (e.g. every twin has a config.py)
    import fresh rather than reusing another twin's cached module.
    """
    twin_dir = str(twin_dir)
    sys.path.insert(0, twin_dir)
    try:
        yield
    finally:
        if twin_dir in sys.path:
            sys.path.remove(twin_dir)
        for name in module_names:
            sys.modules.pop(name, None)


# ---------------------------------------------------------------------------
# Manufacturing
# ---------------------------------------------------------------------------
def run_manufacturing_twin(twin_dir, machine_id: str = "CNC-01", n_readings: int = None) -> dict:
    """
    Replays manufacturing_twin's live_machine_stream.csv through the real
    predictor.py (loads machine_failure_xgboost.json, the actual trained
    XGBoost model) and twin.py's MachineTwin, exactly as dashboard.py does.

    Returns the twin's native evidence_for_dss() dict for the LAST reading
    replayed (i.e. current state) -- pass n_readings to stop early, or leave
    None to replay the whole stream and report the final state.
    """
    twin_dir = Path(twin_dir)
    required = ["stream.py", "twin.py", "predictor.py",
                "models/machine_failure_xgboost.json",
                "models/label_encoder.pkl", "models/feature_columns.pkl",
                "data/live_machine_stream.csv"]
    missing = [f for f in required if not (twin_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Manufacturing twin directory {twin_dir} is missing: {missing}")

    module_names = ["predictor", "twin", "stream"]
    with _isolated_path(twin_dir, module_names):
        for name in module_names:
            sys.modules.pop(name, None)
        import stream
        import twin as twin_module

        machine_twin = twin_module.MachineTwin(machine_id)
        for i, reading in enumerate(stream.replay(delay=0)):
            machine_twin.update(reading)
            if n_readings is not None and i + 1 >= n_readings:
                break

        return machine_twin.evidence_for_dss()


# ---------------------------------------------------------------------------
# Healthcare
# ---------------------------------------------------------------------------
def run_healthcare_twin(twin_dir, hospital_name: str = None) -> list:
    """
    Runs twin_app's real pipeline: data_loader -> hospital_selector ->
    hospital_profile -> demand_intensity -> digital_twin.run_simulation ->
    metrics.summarize_scenario -> forecasting -> dss_interface.build_evidence.

    Uses config.py's own default calibration (TARGET_STEADY_STATE_OCCUPANCY,
    SYSTEM_ADJUSTMENT_TIME_CONSTANT_HOURS) and default facility, i.e. exactly
    what app.py shows before anyone opens the Advanced/Research Settings
    expander. Returns the native list of per-scenario evidence dicts
    (LOW/BASE/HIGH) -- feed one scenario to evidence_schema.adapt_healthcare_evidence().
    """
    twin_dir = Path(twin_dir)
    required = ["config.py", "data_loader.py", "hospital_selector.py", "hospital_profile.py",
                "demand_intensity.py", "digital_twin.py", "metrics.py", "forecasting.py",
                "dss_interface.py", "data/PMC_Hospital_Infrastructure.csv", "data/dataset3.csv"]
    missing = [f for f in required if not (twin_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Healthcare twin directory {twin_dir} is missing: {missing}")

    module_names = ["config", "data_loader", "hospital_selector", "hospital_profile",
                     "demand_intensity", "digital_twin", "metrics", "forecasting", "dss_interface"]
    with _isolated_path(twin_dir, module_names):
        for name in module_names:
            sys.modules.pop(name, None)
        import config
        import data_loader
        import hospital_selector
        import hospital_profile
        import demand_intensity
        import digital_twin
        import metrics
        import forecasting
        import dss_interface

        pmc_df = data_loader.load_pmc_infrastructure()
        ops_df_raw = data_loader.load_operational_stream()

        name = hospital_name or config.DEFAULT_FACILITY
        row = hospital_selector.get_facility(pmc_df, name)
        profile = hospital_profile.build_profile(row)

        target_occ = config.TARGET_STEADY_STATE_OCCUPANCY
        time_constant = config.SYSTEM_ADJUSTMENT_TIME_CONSTANT_HOURS

        demand_df = demand_intensity.compute_simulated_demand(
            ops_df_raw, profile.total_beds, profile.monthly_footfall, target_occ, time_constant)

        summaries = {}
        for scenario_name, pct in config.SCENARIOS.items():
            sim = digital_twin.run_simulation(
                ops_df=demand_df, total_beds=profile.total_beds, doctors=profile.doctors,
                nurses=profile.nurses, midwives=profile.midwives, initial_occupancy_pct=pct,
                time_constant_hours=time_constant,
            )
            sim["hospital_name"] = profile.name
            s = metrics.summarize_scenario(sim)
            s["initial_occupancy"] = pct
            s["scenario"] = scenario_name
            summaries[scenario_name] = s

        forecast_result = forecasting.run_forecasting(demand_df)
        return dss_interface.build_evidence(profile, summaries, forecast_result)


# ---------------------------------------------------------------------------
# Retail / business
# ---------------------------------------------------------------------------
def run_retail_twin(twin_dir, scenario_name: str = "Baseline", demand_pct: int = 0,
                     lead_time_delta: int = 0, n_stores: int = 3, n_items: int = 10,
                     horizon_days: int = 30) -> tuple:
    """
    Runs retail_digital_twin's real pipeline: data_utils.load_train ->
    forecasting.load_model_bundle (the actual trained XGBoost demand model)
    -> simulation.run_scenario (twice: baseline + the requested scenario) ->
    network_aggregate -> business_health_score -> recommendations ->
    simulation.export_dss_evidence. This is the same call sequence app.py
    uses for the "Sample network (fast)" mode with default sidebar values.

    Returns (dss_payload, health_score) for the requested scenario -- feed
    both to evidence_schema.adapt_retail_evidence().
    """
    twin_dir = Path(twin_dir)
    required = ["data_utils.py", "forecasting.py", "twin.py", "simulation.py",
                "recommendations.py", "data/train.csv", "models/demand_model.pkl",
                "models/metrics.json"]
    missing = [f for f in required if not (twin_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Retail twin directory {twin_dir} is missing: {missing}")

    module_names = ["data_utils", "forecasting", "twin", "simulation", "recommendations"]
    with _isolated_path(twin_dir, module_names):
        for name in module_names:
            sys.modules.pop(name, None)
        import json
        import pandas as pd
        from data_utils import load_train
        from forecasting import load_model_bundle
        from twin import ScenarioConfig, SupplierConfig
        import simulation
        import recommendations as rec_engine

        train_df = load_train(str(twin_dir / "data" / "train.csv"))
        model_bundle = load_model_bundle(str(twin_dir / "models" / "demand_model.pkl"))
        with open(twin_dir / "models" / "metrics.json") as f:
            model_metrics = json.load(f)

        sel_stores = sorted(train_df.store.unique().tolist())[:n_stores]
        sel_items = sorted(train_df.item.unique().tolist())[:n_items]
        pairs = tuple((s, i) for s in sel_stores for i in sel_items)

        start_date = train_df.date.max() + pd.Timedelta(days=1)
        base_lead_time, order_coverage_days, service_level_target = 5, 14, 0.95
        demand_mult = 1.0 + demand_pct / 100.0

        common = dict(
            model_bundle=model_bundle, train_df=train_df, pairs=pairs,
            supplier=SupplierConfig(base_lead_time_days=base_lead_time,
                                     order_coverage_days=order_coverage_days),
            start_date=start_date, horizon_days=horizon_days,
        )

        _, base_steps, base_summary = simulation.run_scenario(
            scenario=ScenarioConfig(name="Baseline", demand_multiplier=1.0,
                                     lead_time_change_days=0,
                                     service_level_target=service_level_target),
            **common)
        _, scn_steps, scn_summary = simulation.run_scenario(
            scenario=ScenarioConfig(name=scenario_name, demand_multiplier=demand_mult,
                                     lead_time_change_days=lead_time_delta,
                                     service_level_target=service_level_target),
            **common)

        base_agg = simulation.network_aggregate(base_summary)
        scn_agg = simulation.network_aggregate(scn_summary)
        comparison = simulation.compare_baseline_vs_scenario(base_agg, scn_agg)
        health_base = simulation.business_health_score(base_agg)
        health_scn = simulation.business_health_score(scn_agg)
        critical_df = simulation.rank_critical_entities(scn_summary, top_n=10)
        recs = rec_engine.build_recommendations(
            base_agg, scn_agg, comparison,
            ScenarioConfig(name=scenario_name, demand_multiplier=demand_mult,
                            lead_time_change_days=lead_time_delta,
                            service_level_target=service_level_target),
            critical_df, health_base, health_scn,
        )

        (twin_dir / "outputs").mkdir(exist_ok=True)
        dss_payload = simulation.export_dss_evidence(
            path=str(twin_dir / "outputs" / "dss_evidence.json"),
            model_metrics=model_metrics, baseline_agg=base_agg, scenario_agg=scn_agg,
            scenario_config=ScenarioConfig(name=scenario_name, demand_multiplier=demand_mult,
                                            lead_time_change_days=lead_time_delta,
                                            service_level_target=service_level_target),
            comparison=comparison, critical_df=critical_df, recommendations=recs,
            n_stores=len(sel_stores), n_items=len(sel_items), horizon_days=horizon_days,
        )

        return dss_payload, health_scn


if __name__ == "__main__":
    import json

    MANUFACTURING_DIR = Path(__file__).parent / "manufacturing_twin"
    HEALTHCARE_DIR = Path(__file__).parent / "healthcare_twin"
    RETAIL_DIR = Path(__file__).parent / "retail_twin"

    print("=== Manufacturing (live) ===")
    print(json.dumps(run_manufacturing_twin(MANUFACTURING_DIR), indent=2))

    print("\n=== Healthcare (live) ===")
    print(json.dumps(run_healthcare_twin(HEALTHCARE_DIR), indent=2, default=str)[:800])

    print("\n=== Retail (live) ===")
    payload, score = run_retail_twin(RETAIL_DIR)
    print("health_score:", score)
