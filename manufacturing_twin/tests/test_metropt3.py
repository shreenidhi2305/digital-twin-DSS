"""Parity and contract tests for the MetroPT-3 SAE manufacturing twin."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TWIN_DIR = Path(__file__).resolve().parents[1]
if str(TWIN_DIR) not in sys.path:
    sys.path.insert(0, str(TWIN_DIR))

from dt.episodes import merge_alarm_episodes, model_may_be_drifting
from dt.features.metropt3 import (
    DUTY_WINDOW,
    FEATURE_NAMES,
    RAW_SIGNALS,
    SEGMENT_GAP_S,
    MetroFeatureStream,
    build_feature_frame,
    operating_state,
)
from dt.models.metropt3_sae import (
    PRIMARY_BUDGET,
    ArtifactChecksumError,
    MetroSAEDetector,
    verify_manifest,
)

MODEL_DIR = TWIN_DIR / "models" / "metropt3_sae"
REPO_ROOT = TWIN_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _raw_row(ts, **overrides):
    row = {"timestamp": ts}
    for i, name in enumerate(RAW_SIGNALS):
        row[name] = float(i + 1)
    row.update(overrides)
    return row


def _regular_frame(n, start="2020-02-01 00:00:00", step_s=10, gap_at=None, gap_s=60):
    start_ts = pd.Timestamp(start)
    rows = []
    cursor = start_ts
    for i in range(n):
        if gap_at is not None and i == gap_at:
            cursor = cursor + pd.Timedelta(seconds=gap_s)
        row = _raw_row(cursor, TP3=10.0 + 0.01 * i, DV_eletric=1.0 if i % 5 == 0 else 0.0,
                       COMP=0.0 if i % 5 == 0 else 1.0, Motor_current=7.0 if i % 5 == 0 else 4.0)
        rows.append(row)
        cursor = cursor + pd.Timedelta(seconds=step_s)
    return pd.DataFrame(rows)


def test_manifest_matches_artifacts():
    verify_manifest(MODEL_DIR)


def test_manifest_mismatch_is_loud(tmp_path):
    for path in MODEL_DIR.iterdir():
        target = tmp_path / path.name
        target.write_bytes(path.read_bytes())
    weights = tmp_path / "sae_model_state.pt"
    data = bytearray(weights.read_bytes())
    data[-1] = (data[-1] + 1) % 256
    weights.write_bytes(data)
    with pytest.raises(ArtifactChecksumError, match="Checksum mismatch"):
        verify_manifest(tmp_path)


def test_detector_loads_and_rejects_bad_columns():
    detector = MetroSAEDetector.load(MODEL_DIR)
    assert detector.threshold("1.0") == pytest.approx(0.023753321729600516)
    assert detector.threshold(1) == detector.threshold("1.0")
    assert list(detector.feature_names) == list(FEATURE_NAMES)
    frame = pd.DataFrame(np.zeros((2, len(FEATURE_NAMES))), columns=list(FEATURE_NAMES))
    scores = detector.score(frame)
    assert scores.shape == (2,)
    assert np.isfinite(scores).all()
    contrib = detector.feature_contributions(frame)
    assert contrib.shape == (2, 23)
    swapped = frame[list(reversed(frame.columns))]
    with pytest.raises(ValueError, match="same order"):
        detector.score(swapped)


def test_nan_rows_are_not_imputed():
    detector = MetroSAEDetector.load(MODEL_DIR)
    frame = pd.DataFrame(np.ones((3, len(FEATURE_NAMES))), columns=list(FEATURE_NAMES))
    frame.loc[1, "TP2"] = np.nan
    scores = detector.score(frame)
    assert np.isfinite(scores[0]) and np.isfinite(scores[2])
    assert np.isnan(scores[1])
    assert detector.is_alarm(scores[1], "1.0") is False
    # A finite neighbour is unchanged by the NaN row sitting between them.
    alone = detector.score(frame.iloc[[0]])
    assert scores[0] == pytest.approx(alone[0])


def test_full_window_and_segment_gap():
    raw = _regular_frame(DUTY_WINDOW + 5, gap_at=None)
    featured = build_feature_frame(raw)
    assert featured["feat_valid"].iloc[: DUTY_WINDOW - 1].eq(False).all()
    assert bool(featured["feat_valid"].iloc[DUTY_WINDOW - 1])
    assert pd.isna(featured["d_TP3_per_s"].iloc[0])
    assert featured["tp3_minus_h1"].iloc[10] == pytest.approx(
        featured["TP3"].iloc[10] - featured["H1"].iloc[10]
    )

    gapped = _regular_frame(DUTY_WINDOW + 10, gap_at=DUTY_WINDOW, gap_s=SEGMENT_GAP_S + 5)
    gapped_features = build_feature_frame(gapped)
    # The first row after the gap is a new segment and is not valid yet.
    after = gapped_features.iloc[DUTY_WINDOW]
    assert bool(after["feat_valid"]) is False
    assert pd.isna(after["load_duty_30m"])


def test_no_future_leakage():
    raw = _regular_frame(DUTY_WINDOW + 30)
    prefix = build_feature_frame(raw.iloc[: DUTY_WINDOW + 5])
    full = build_feature_frame(raw)
    cols = list(FEATURE_NAMES)
    pd.testing.assert_frame_equal(
        prefix[cols].reset_index(drop=True),
        full.iloc[: len(prefix)][cols].reset_index(drop=True),
    )


def test_batch_matches_stream_on_10s_and_1hz():
    for step in (10, 1):
        raw = _regular_frame(DUTY_WINDOW + 40 if step == 10 else 400, step_s=step, gap_at=50, gap_s=45)
        batch = build_feature_frame(raw)
        stream = MetroFeatureStream()
        rows = []
        for record in raw.to_dict(orient="records"):
            rows.extend(stream.push_raw(record))
        rows.extend(stream.flush())
        streamed = pd.DataFrame(rows)
        assert len(streamed) == len(batch)
        for column in FEATURE_NAMES:
            left = streamed[column].to_numpy(dtype=float)
            right = batch[column].to_numpy(dtype=float)
            assert np.allclose(left, right, atol=1e-8, rtol=1e-6, equal_nan=True), column
        assert streamed["feat_valid"].tolist() == batch["feat_valid"].tolist()


def test_operating_state_uses_current_and_comp():
    assert operating_state(0.2, 1) == "off"
    assert operating_state(7.0, 0.0) == "load"
    assert operating_state(4.0, 1.0) == "offload"
    assert operating_state(np.nan, 1.0) is None


def test_episode_merge_is_thirty_minutes():
    base = pd.Timestamp("2020-06-05 10:00:00")
    times = [base, base + pd.Timedelta(minutes=20), base + pd.Timedelta(minutes=51)]
    scores = np.array([0.03, 0.05, 0.04])
    alarm = np.array([True, True, True])
    contrib = np.tile(np.arange(len(FEATURE_NAMES), dtype=float), (3, 1))
    episodes = merge_alarm_episodes(times, alarm, scores, contrib, list(FEATURE_NAMES), "1.0", 30)
    assert len(episodes) == 2
    assert episodes[0]["n_rows"] == 2
    assert episodes[0]["peak_score"] == pytest.approx(0.05)
    assert episodes[0]["end"].startswith("2020-06-05 10:20:00")
    assert episodes[1]["n_rows"] == 1
    assert model_may_be_drifting(2.1, "1.0") is True
    assert model_may_be_drifting(2.0, "1.0") is False
    assert model_may_be_drifting(None, "1.0") is False


def test_run_config_windows_match_code():
    config = json.loads((MODEL_DIR / "run_config.json").read_text(encoding="utf-8"))["config"]
    assert config["segment_gap_s"] == SEGMENT_GAP_S
    assert config["merge_minutes"] == 30
    assert config["primary_budget"] == float(PRIMARY_BUDGET)
    names = json.loads((MODEL_DIR / "feature_names.json").read_text(encoding="utf-8"))
    assert names == list(FEATURE_NAMES)


def _metro_csv():
    from stream import DATA_CANDIDATES, DATA_DIR
    for name in DATA_CANDIDATES:
        path = DATA_DIR / name
        if path.is_file():
            return path
    return None


def test_seven_day_score_distribution_or_skip():
    """
    Exact parity against features.parquet is not possible: that file was not
    in the artifact bundle. When the reference CSV is present, check that a
    7-day score distribution sits near the published monthly drift table.
    """
    csv_path = _metro_csv()
    if csv_path is None:
        pytest.skip("MetroPT3 CSV is not on disk; exact parity was not verified")

    from stream import load_window
    from twin import MachineTwin

    raw = load_window(start="2020-02-01", end="2020-02-08", path=csv_path)
    assert len(raw) > 1000, "7-day slice was unexpectedly short"
    twin = MachineTwin(detector=MetroSAEDetector.load(MODEL_DIR))
    twin.replay_frame(raw)
    scores = np.array([s for s in twin._scores if np.isfinite(s)], dtype=float)
    assert len(scores) > 100
    median = float(np.median(scores))
    p95 = float(np.quantile(scores, 0.95))
    # score_drift_monthly.csv: medians ~1e-5; April-May p95 ~1e-2, later months ~1e-4.
    assert 1e-6 <= median <= 1e-4, f"median score {median} is outside the published ~1e-5 band"
    assert 1e-5 <= p95 <= 5e-2, f"p95 score {p95} is outside the published drift range"
    observed_days = (
        pd.Timestamp(twin.history[-1]["timestamp"]) - pd.Timestamp(twin.history[0]["timestamp"])
    ).total_seconds() / 86400
    alarms = sum(1 for ep in twin.episodes)
    rate = alarms / observed_days if observed_days else float("inf")
    # Budgets are 0.5/1/2 episodes per day on validation. A quiet February week
    # should not be orders of magnitude above the looser budget.
    assert rate < 10, f"alarm rate {rate:.2f}/day is far above the frozen budgets"


def test_evidence_packet_has_no_failure_probability():
    from twin import create_twin
    raw = _regular_frame(DUTY_WINDOW + 2)
    twin = create_twin(detector=MetroSAEDetector.load(MODEL_DIR))
    twin.replay_frame(raw)
    assert twin.history[10]["data_quality"] == ["incomplete_window"]
    assert twin.history[-1]["data_quality"] == []
    packet = twin.evidence_for_dss()
    assert "failure_probability" not in packet
    assert "tool_wear" not in packet
    assert packet["score_kind"] == "reconstruction_mse"
    assert packet["budget"] == "1.0"
    assert "anomaly_score" in packet

    from evidence_schema import adapt_manufacturing_evidence
    adapted = adapt_manufacturing_evidence(packet)
    assert adapted["operational_score"] is None
    assert "failure_probability" not in adapted["key_metrics"]
    assert "tool_wear_min" not in adapted["key_metrics"]
    assert "anomaly_score" in adapted["key_metrics"]
