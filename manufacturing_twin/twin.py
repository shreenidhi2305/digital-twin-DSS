"""
Digital twin of one MetroPT-3 air compressor (APU).

Each closed 10 s bin is scored with the frozen sparse autoencoder. The twin
exposes an anomaly score, the frozen threshold, an alarm flag, the active
episode, the top contributing features, the operating state, and data-quality
flags. It does not estimate failure probability or remaining useful life.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
from dt.episodes import (
    MERGE_MINUTES,
    active_episode,
    alarm_rate_per_day,
    model_may_be_drifting,
    top_features,
)
from dt.features.metropt3 import (
    FEATURE_NAMES,
    MetroFeatureStream,
    build_feature_frame,
    operating_state,
)
from dt.models.metropt3_sae import PRIMARY_BUDGET, MetroSAEDetector


def _quality_flags(row: dict) -> list[str]:
    """Warmup NaNs are an incomplete window. non_finite is only for bad raw sensors."""
    raw_names = (
        "TP2", "TP3", "H1", "DV_pressure", "Oil_temperature",
        "Motor_current", "COMP", "DV_eletric",
    )
    flags = []
    raw_missing = False
    for name in raw_names:
        value = row.get(name)
        if value is None or not np.isfinite(value):
            raw_missing = True
            break
    if raw_missing:
        flags.append("non_finite_feature")
    if not bool(row.get("feat_valid", False)):
        flags.append("incomplete_window")
    return flags


def _recommendation(alarm: bool, quality: list[str]) -> str:
    if "incomplete_window" in quality or "non_finite_feature" in quality:
        return (
            "No anomaly score yet. A full 30-minute trailing window is required, "
            "and rows with missing sensors are flagged rather than filled in."
        )
    if alarm:
        return (
            "Anomaly alarm: reconstruction error is at or above the frozen threshold "
            "for the selected false-alarm budget. This is not a failure prediction "
            "and not a remaining-useful-life estimate. Compare the top contributing "
            "sensors with the reported episode."
        )
    return (
        "No anomaly alarm at the selected budget. A score under the threshold "
        "does not certify that the compressor is healthy."
    )


class MachineTwin:
    """Live mirror of one APU compressor. `budget` selects a frozen threshold."""

    def __init__(self, machine_id: str = "APU-01", budget: str = PRIMARY_BUDGET,
                 model_dir=None, detector: MetroSAEDetector | None = None):
        self.machine_id = machine_id
        self.detector = detector if detector is not None else MetroSAEDetector.load(model_dir)
        self.budget = self._check_budget(budget)
        self.features = MetroFeatureStream()
        self.latest_state: dict | None = None
        self.history: list[dict] = []
        self.episodes: list[dict] = []
        self._scores: list[float] = []
        self._contrib: list[np.ndarray] = []

    def _check_budget(self, budget: str) -> str:
        self.detector.threshold(budget)
        text = str(budget).strip()
        try:
            return f"{float(text):.1f}"
        except ValueError:
            return text

    def set_budget(self, budget: str) -> dict | None:
        """Switch the frozen threshold and rebuild episodes. Does not rescore."""
        self.budget = self._check_budget(budget)
        self.episodes = []
        self._resnapshot_history()
        if self.history:
            self.latest_state = self.history[-1]
        return self.latest_state

    def update(self, reading: dict) -> dict | None:
        """Push one raw sensor row. Returns the newest closed-bin state, if a bin closed."""
        emitted = self.features.push_raw(reading)
        state = None
        for row in emitted:
            state = self._ingest_feature_row(row)
        return state if state is not None else self.latest_state

    def flush(self) -> dict | None:
        state = None
        for row in self.features.flush():
            state = self._ingest_feature_row(row)
        return state if state is not None else self.latest_state

    def replay_frame(self, raw: pd.DataFrame) -> dict | None:
        """
        Score a raw frame with the batch feature path (same definition as the
        stream; covered by the parity test) and replace this twin's history.
        """
        featured = build_feature_frame(raw)
        self.reset_history()
        if featured.empty:
            return None
        model_input = featured.loc[:, list(FEATURE_NAMES)]
        scores = self.detector.score(model_input)
        contrib = self.detector.feature_contributions(model_input)
        for i in range(len(featured)):
            self._append_scored(featured.iloc[i].to_dict(), float(scores[i]), contrib[i])
        self.latest_state = self.history[-1] if self.history else None
        return self.latest_state

    def reset_history(self) -> None:
        self.features.reset()
        self.latest_state = None
        self.history = []
        self.episodes = []
        self._scores = []
        self._contrib = []

    def _ingest_feature_row(self, row: dict) -> dict:
        frame = pd.DataFrame([{name: row.get(name) for name in FEATURE_NAMES}])
        score = float(self.detector.score(frame)[0])
        contrib = self.detector.feature_contributions(frame)[0]
        self._append_scored(row, score, contrib)
        self.latest_state = self.history[-1]
        return self.latest_state

    def _append_scored(self, row: dict, score: float, contrib: np.ndarray) -> None:
        quality = _quality_flags(row)
        finite = bool(np.isfinite(score))
        alarm = bool(self.detector.is_alarm(score, self.budget)) if finite else False
        state = self._state(row, score if finite else None, contrib, alarm, quality)
        self.history.append(state)
        self._scores.append(score)
        self._contrib.append(np.asarray(contrib, dtype=np.float64))
        self._extend_episodes(len(self.history) - 1)
        self._snapshot_row(len(self.history) - 1)

    def _state(self, row: dict, score, contrib: np.ndarray, alarm: bool, quality: list[str]) -> dict:
        threshold = self.detector.threshold(self.budget)
        return {
            "timestamp": pd.Timestamp(row["timestamp"]).isoformat(sep=" "),
            "ingested_at": datetime.now(UTC).isoformat(),
            "domain": "Manufacturing",
            "machine": self.machine_id,
            "anomaly_score": None if score is None or not np.isfinite(score) else float(score),
            "threshold": threshold,
            "budget": self.budget,
            "alarm": alarm,
            "operating_state": operating_state(row.get("Motor_current"), row.get("COMP")),
            "top_features": top_features(contrib, list(FEATURE_NAMES), k=3) if score is not None else [],
            "data_quality": quality,
            "motor_current_A": _finite_or_none(row.get("Motor_current")),
            "comp": _finite_or_none(row.get("COMP")),
            "dv_eletric": _finite_or_none(row.get("DV_eletric")),
            "tp2_bar": _finite_or_none(row.get("TP2")),
            "tp3_bar": _finite_or_none(row.get("TP3")),
            "oil_temperature_C": _finite_or_none(row.get("Oil_temperature")),
            "recommendation": _recommendation(alarm, quality),
            "feat_valid": "incomplete_window" not in quality and "non_finite_feature" not in quality,
        }

    def _extend_episodes(self, index: int) -> None:
        """Add row `index` using only alarms seen so far (no future rows)."""
        score = self._scores[index]
        if not (np.isfinite(score) and self.detector.is_alarm(score, self.budget)):
            return
        ts = pd.Timestamp(self.history[index]["timestamp"])
        names = top_features(self._contrib[index], list(FEATURE_NAMES))
        window = pd.Timedelta(minutes=float(self.detector.merge_minutes or MERGE_MINUTES))
        if self.episodes and ts - pd.Timestamp(self.episodes[-1]["end"]) <= window:
            ep = self.episodes[-1]
            ep["end"] = ts.isoformat(sep=" ")
            ep["n_rows"] += 1
            if score > ep["peak_score"]:
                ep["peak_score"] = float(score)
                ep["top_features"] = names
        else:
            self.episodes.append({
                "start": ts.isoformat(sep=" "),
                "end": ts.isoformat(sep=" "),
                "n_rows": 1,
                "peak_score": float(score),
                "top_features": names,
                "budget": self.budget,
            })

    def _snapshot_row(self, index: int) -> None:
        row = self.history[index]
        score = self._scores[index]
        finite = bool(np.isfinite(score))
        alarm = bool(self.detector.is_alarm(score, self.budget)) if finite else False
        row["threshold"] = self.detector.threshold(self.budget)
        row["budget"] = self.budget
        row["alarm"] = alarm
        row["anomaly_score"] = float(score) if finite else None
        row["recommendation"] = _recommendation(alarm, row["data_quality"])
        merge = self.detector.merge_minutes or MERGE_MINUTES
        current = active_episode(self.episodes, row["timestamp"], merge)
        if current is None:
            row["active_episode"] = None
        else:
            # Copy so a later alarm cannot rewrite this row's episode view.
            row["active_episode"] = {**current, "top_features": list(current["top_features"])}
        # Drift uses episodes opened up to this row. Rebuilding the rate from
        # the full episode list is causal because episodes are appended in order
        # and a later call does not rewrite earlier rows during replay.
        row["model_may_be_drifting"] = model_may_be_drifting(
            alarm_rate_per_day(
                self.episodes,
                self.history[0]["timestamp"],
                row["timestamp"],
            ),
            self.budget,
        )

    def _resnapshot_history(self) -> None:
        """Used when the budget changes: episodes are recomputed from stored scores."""
        self.episodes = []
        for i in range(len(self.history)):
            self._extend_episodes(i)
            self._snapshot_row(i)

    def evidence_for_dss(self) -> dict:
        """Compact packet for the DSS. Scores are anomaly alarms, not failure probabilities."""
        if self.latest_state is None:
            return {}
        s = self.latest_state
        episode = s.get("active_episode")
        return {
            "domain": "Manufacturing",
            "machine": s["machine"],
            "timestamp": s["timestamp"],
            "alarm": s["alarm"],
            "anomaly_score": s["anomaly_score"],
            "threshold": s["threshold"],
            "budget": s["budget"],
            "operating_state": s["operating_state"],
            "top_features": s["top_features"],
            "active_episode": episode,
            "data_quality": s["data_quality"],
            "model_may_be_drifting": s["model_may_be_drifting"],
            "recommendation": s["recommendation"],
            "score_kind": "reconstruction_mse",
        }


def _finite_or_none(value):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def create_twin(machine_id: str = "APU-01", **kwargs) -> MachineTwin:
    """Build the MetroPT-3 air-compressor twin."""
    return MachineTwin(machine_id=machine_id, **kwargs)
