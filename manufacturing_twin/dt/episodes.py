"""Merge anomaly alarms into episodes. A gap of 30 minutes or less stays one episode."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

MERGE_MINUTES = 30.0


def _as_timestamp(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return ts


def top_features(contributions: np.ndarray, feature_names: list[str], k: int = 3) -> list[str]:
    """Names of the k largest per-feature squared errors. Ties break on feature order."""
    row = np.asarray(contributions, dtype=np.float64)
    if row.shape != (len(feature_names),) or not np.isfinite(row).any():
        return []
    order = np.argsort(np.where(np.isfinite(row), -row, np.inf), kind="mergesort")
    names = []
    for idx in order:
        if not np.isfinite(row[idx]):
            continue
        names.append(feature_names[int(idx)])
        if len(names) >= k:
            break
    return names


def merge_alarm_episodes(
    timestamps,
    is_alarm,
    scores,
    contributions: np.ndarray,
    feature_names: list[str],
    budget: str,
    merge_minutes: float = MERGE_MINUTES,
) -> list[dict]:
    """
    Walk alarm rows in time order. A new alarm extends the current episode
    when it falls within `merge_minutes` of the previous alarm timestamp.
    n_rows counts alarm rows only. top_features are taken from the peak score.
    """
    times = [_as_timestamp(t) for t in timestamps]
    alarm = np.asarray(is_alarm, dtype=bool)
    score = np.asarray(scores, dtype=np.float64)
    if len(times) != len(alarm) or len(times) != len(score):
        raise ValueError("timestamps, is_alarm, and scores must have the same length")
    if contributions.shape[0] != len(times):
        raise ValueError("contributions row count does not match timestamps")

    window = timedelta(minutes=float(merge_minutes))
    episodes: list[dict] = []
    for i, ts in enumerate(times):
        if not alarm[i] or not np.isfinite(score[i]):
            continue
        names = top_features(contributions[i], feature_names)
        if episodes and ts - episodes[-1]["_last"] <= window:
            ep = episodes[-1]
            ep["end"] = ts.isoformat(sep=" ")
            ep["n_rows"] += 1
            ep["_last"] = ts
            if score[i] > ep["peak_score"]:
                ep["peak_score"] = float(score[i])
                ep["top_features"] = names
        else:
            episodes.append({
                "start": ts.isoformat(sep=" "),
                "end": ts.isoformat(sep=" "),
                "n_rows": 1,
                "peak_score": float(score[i]),
                "top_features": names,
                "budget": str(budget),
                "_last": ts,
            })
    for ep in episodes:
        ep.pop("_last", None)
    return episodes


def active_episode(episodes: list[dict], at, merge_minutes: float = MERGE_MINUTES) -> dict | None:
    """The episode still open at `at`: its latest alarm is within the merge window."""
    if not episodes:
        return None
    moment = _as_timestamp(at)
    window = timedelta(minutes=float(merge_minutes))
    last = episodes[-1]
    end = _as_timestamp(last["end"])
    start = _as_timestamp(last["start"])
    if start <= moment and moment - end <= window:
        return last
    return None


def alarm_rate_per_day(episodes: list[dict], history_start, history_end, days: float = 7.0) -> float | None:
    """
    Episodes whose start falls in the last `days` calendar days, divided by `days`.
    Returns None until the scored history itself covers that span.
    """
    if history_start is None or history_end is None:
        return None
    start = _as_timestamp(history_start)
    end = _as_timestamp(history_end)
    if end - start < timedelta(days=float(days)):
        return None
    window_start = end - timedelta(days=float(days))
    count = 0
    for ep in episodes:
        ep_start = _as_timestamp(ep["start"])
        if window_start < ep_start <= end:
            count += 1
    return count / float(days)


def model_may_be_drifting(rate_per_day: float | None, budget: str | float) -> bool:
    """True when the 7-day episode rate is more than twice the selected budget."""
    if rate_per_day is None:
        return False
    return float(rate_per_day) > 2.0 * float(budget)
