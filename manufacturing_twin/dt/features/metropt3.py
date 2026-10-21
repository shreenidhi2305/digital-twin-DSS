"""
Trailing-only features for the MetroPT-3 sparse autoencoder.

The published MetroPT-3 file is already one row every 10 seconds. Faster
frames are averaged into non-overlapping 10 s bins. Windows never look
ahead, and a gap longer than 30 s starts a new segment so a window cannot
cross a hole in the log.

Rows that do not yet have a full 30-minute trailing window (or a previous
sample for the TP3 derivative) are marked feat_valid=False. They are not
imputed and must not be scored.

Assumptions, checked against the frozen scaler rather than a missing
notebook (see manufacturing_twin/README.md):

- load_duty_30m is the trailing 180-sample mean of DV_eletric. The MetroPT-3
  description defines DV_eletric as active while the compressor is under
  load, and the frozen training mean of this feature is ~0.121, the same
  location as DV_eletric. COMP is the opposite signal (active when off or
  offloaded) and is not the duty numerator.
- d_TP3_per_s is (TP3 - previous TP3) / elapsed seconds inside a segment.
- 5-minute mean and standard deviation use 30 trailing samples. Standard
  deviation is the sample statistic (ddof=1), which is pandas' default.
- Operating state is a display label, not a model input. Motor_current below
  2 A is "off" (the description's off state is near 0 A and offload is near
  4 A). Otherwise COMP < 0.5 is "load", because COMP is active when the unit
  is off or offloaded. The remaining case is "offload".
"""

from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

RAW_SIGNALS = (
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
)

ROLL_COLUMNS = (
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Oil_temperature",
    "Motor_current",
)

FEATURE_NAMES = (
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Oil_temperature",
    "Motor_current",
    "COMP",
    "DV_eletric",
    "tp3_minus_h1",
    "d_TP3_per_s",
    "load_duty_30m",
    "TP2_mean_5m",
    "TP2_std_5m",
    "TP3_mean_5m",
    "TP3_std_5m",
    "H1_mean_5m",
    "H1_std_5m",
    "DV_pressure_mean_5m",
    "DV_pressure_std_5m",
    "Oil_temperature_mean_5m",
    "Oil_temperature_std_5m",
    "Motor_current_mean_5m",
    "Motor_current_std_5m",
)

BIN_SECONDS = 10.0
SEGMENT_GAP_S = 30.0
MEAN_WINDOW = 30       # 5 min at 10 s
DUTY_WINDOW = 180      # 30 min at 10 s
STD_DDOF = 1
OFF_CURRENT_A = 2.0
TIMESTAMP_CANDIDATES = ("timestamp", "Timestamp", "time", "Time", "datetime")


def operating_state(motor_current: float, comp: float) -> str | None:
    """load / offload / off. None when either sensor is missing."""
    if not np.isfinite(motor_current) or not np.isfinite(comp):
        return None
    if motor_current < OFF_CURRENT_A:
        return "off"
    if comp < 0.5:
        return "load"
    return "offload"


def _as_frame(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy()
    if "timestamp" not in frame.columns:
        for name in TIMESTAMP_CANDIDATES:
            if name in frame.columns:
                frame = frame.rename(columns={name: "timestamp"})
                break
        else:
            raise ValueError(
                "Raw frame needs a timestamp column. "
                f"Looked for {TIMESTAMP_CANDIDATES}. Columns: {list(raw.columns)}"
            )
    missing = [c for c in RAW_SIGNALS if c not in frame.columns]
    if missing:
        raise ValueError(f"Raw frame is missing MetroPT-3 signals: {missing}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if frame["timestamp"].isna().any():
        raise ValueError("Raw frame has timestamps that could not be parsed. They were not filled in.")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp", kind="mergesort")
    frame = frame.drop_duplicates("timestamp", keep="last")
    for column in RAW_SIGNALS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.reset_index(drop=True)


def _segment_ids(timestamps: pd.Series) -> np.ndarray:
    gap = timestamps.diff().dt.total_seconds()
    new_segment = gap.isna() | (gap > SEGMENT_GAP_S)
    return new_segment.cumsum().to_numpy()


def bins_from_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """
    10 s bins plus a segment id. A median gap of 8–12 s is left as-is
    (MetroPT-3 is published at 0.1 Hz). Faster data is averaged inside each
    segment. Empty bins are not created.
    """
    frame = _as_frame(raw)
    if frame.empty:
        columns = ["timestamp", "segment_id", *RAW_SIGNALS]
        return pd.DataFrame(columns=columns)

    frame["segment_id"] = _segment_ids(frame["timestamp"])
    pieces = []
    for segment_id, group in frame.groupby("segment_id", sort=False):
        origin = group["timestamp"].iloc[0]
        rel_s = (group["timestamp"] - origin).dt.total_seconds().to_numpy()
        # Floor in integer seconds so a 10.0 s step is the next bin, including
        # when the datetime resolution is microseconds rather than nanoseconds.
        bin_id = np.floor(rel_s / BIN_SECONDS + 1e-9).astype(np.int64)
        grouped = group.assign(_bin=bin_id).groupby("_bin", sort=True)
        # A missing sample must flag the bin. GroupBy.mean(skipna=False) is
        # missing on some pandas builds the dashboard runs under, so divide
        # the sum by the full bin length and blank any column that dropped a row.
        block = grouped[list(RAW_SIGNALS)]
        size = grouped.size()
        means = block.sum(min_count=1).div(size, axis=0)
        means = means.where(block.count().eq(size, axis=0))
        means.insert(0, "timestamp", grouped["timestamp"].max().to_numpy())
        means.insert(1, "segment_id", int(segment_id))
        pieces.append(means.reset_index(drop=True))
    return pd.concat(pieces, ignore_index=True)


def features_from_bins(bins: pd.DataFrame) -> pd.DataFrame:
    """23 model features from an already-binned 10 s frame. Trailing windows only."""
    if bins.empty:
        columns = ["timestamp", "segment_id", "feat_valid", *FEATURE_NAMES]
        return pd.DataFrame(columns=columns)

    pieces = []
    ordered = bins.sort_values(["segment_id", "timestamp"], kind="mergesort")
    for segment_id, group in ordered.groupby("segment_id", sort=False):
        g = group.copy().reset_index(drop=True)
        dt = g["timestamp"].diff().dt.total_seconds()
        dt = dt.where(dt > 0)
        g["tp3_minus_h1"] = g["TP3"] - g["H1"]
        g["d_TP3_per_s"] = g["TP3"].diff() / dt
        g["load_duty_30m"] = g["DV_eletric"].rolling(DUTY_WINDOW, min_periods=DUTY_WINDOW).mean()
        for column in ROLL_COLUMNS:
            window = g[column].rolling(MEAN_WINDOW, min_periods=MEAN_WINDOW)
            g[f"{column}_mean_5m"] = window.mean()
            g[f"{column}_std_5m"] = window.std(ddof=STD_DDOF)
        g["feat_valid"] = g[list(FEATURE_NAMES)].notna().all(axis=1).to_numpy()
        g["segment_id"] = segment_id
        pieces.append(g)
    out = pd.concat(pieces, ignore_index=True)
    keep = ["timestamp", "segment_id", "feat_valid", *FEATURE_NAMES]
    return out[keep]


def build_feature_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Batch path: raw sensor frame -> 23 features and feat_valid."""
    return features_from_bins(bins_from_raw(raw))


class MetroFeatureStream:
    """
    Incremental twin of build_feature_frame.

    Push one raw sample at a time. A 10 s bin is emitted when the next bin
    (or a segment gap, or flush()) closes it. Feature rows for the latest
    bin are computed from a ring buffer of the last 180 bins in the segment,
    which is exactly the history the 30-minute window uses.
    """

    def __init__(self):
        self._reset_segment()
        self._last_raw_ts = None
        self._segment_id = 0
        self._origin = None
        self._open_bin = None
        self._open_sum = None
        self._open_count = 0
        self._open_last_ts = None

    def _reset_segment(self) -> None:
        self._buffer: deque[dict] = deque(maxlen=DUTY_WINDOW)

    def _close_bin(self) -> dict | None:
        if self._open_count <= 0:
            return None
        values = self._open_sum / self._open_count
        row = {
            "timestamp": self._open_last_ts,
            "segment_id": self._segment_id,
        }
        for i, name in enumerate(RAW_SIGNALS):
            row[name] = float(values[i])
        self._open_bin = None
        self._open_sum = None
        self._open_count = 0
        self._open_last_ts = None
        self._buffer.append(row)
        featured = features_from_bins(pd.DataFrame(list(self._buffer)))
        return featured.iloc[-1].to_dict()

    def _start_bin(self, bin_id: int, ts: pd.Timestamp, values: np.ndarray) -> None:
        self._open_bin = int(bin_id)
        self._open_sum = values.astype(np.float64).copy()
        self._open_count = 1
        self._open_last_ts = ts

    def push_raw(self, reading: dict) -> list[dict]:
        """Ingest one raw row. Returns zero or one closed feature row."""
        frame = _as_frame(pd.DataFrame([reading]))
        if frame.empty:
            return []
        ts = pd.Timestamp(frame["timestamp"].iloc[0])
        values = frame.loc[frame.index[0], list(RAW_SIGNALS)].to_numpy(dtype=np.float64)
        emitted: list[dict] = []

        if self._last_raw_ts is None:
            self._segment_id = 1
            self._origin = ts
            self._start_bin(0, ts, values)
            self._last_raw_ts = ts
            return emitted

        gap = (ts - self._last_raw_ts).total_seconds()
        if gap < 0:
            raise ValueError("Streaming readings must arrive in timestamp order")
        if gap > SEGMENT_GAP_S:
            closed = self._close_bin()
            if closed is not None:
                emitted.append(closed)
            self._segment_id += 1
            self._origin = ts
            self._reset_segment()
            self._start_bin(0, ts, values)
            self._last_raw_ts = ts
            return emitted

        rel_s = (ts - self._origin).total_seconds()
        bin_id = int(np.floor(rel_s / BIN_SECONDS + 1e-9))
        if self._open_bin is None:
            self._start_bin(bin_id, ts, values)
        elif bin_id == self._open_bin:
            self._open_sum = self._open_sum + values
            self._open_count += 1
            self._open_last_ts = ts
        else:
            closed = self._close_bin()
            if closed is not None:
                emitted.append(closed)
            self._start_bin(bin_id, ts, values)
        self._last_raw_ts = ts
        return emitted

    def flush(self) -> list[dict]:
        """Emit the bin that is still open at the end of a finite replay."""
        closed = self._close_bin()
        return [closed] if closed is not None else []

    def reset(self) -> None:
        self.__init__()
