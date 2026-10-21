"""
Replay MetroPT-3 air-compressor readings in timestamp order.

The manufacturing twin used to replay a synthetic AI4I-style CSV. The live
loop now walks the MetroPT-3 reference file. That file is not training data
and is not shipped in git (it is about 200 MB). Place it at:

    manufacturing_twin/data/MetroPT3_AirCompressor_.csv

The UCI file is often named MetroPT3(AirCompressor).csv; that name is accepted
too. Playback speed is a delay between yielded rows, for demos.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_CANDIDATES = (
    "MetroPT3_AirCompressor_.csv",
    "MetroPT3(AirCompressor).csv",
    "MetroPT3_AirCompressor.csv",
    "MetroPT3.csv",
)


def resolve_data_path(path: Path | None = None) -> Path:
    if path is not None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"MetroPT-3 file not found: {path}")
        return path
    for name in DATA_CANDIDATES:
        candidate = DATA_DIR / name
        if candidate.is_file():
            return candidate
    looked = ", ".join(str(DATA_DIR / name) for name in DATA_CANDIDATES)
    raise FileNotFoundError(
        "MetroPT-3 reference CSV is not in the repo. Download MetroPT-3 from "
        "the UCI repository (dataset 791) and save it as one of: " + looked
    )


def load_window(start=None, end=None, path: Path | None = None,
                max_rows: int | None = None) -> pd.DataFrame:
    """
    Read the reference CSV and keep rows in [start, end). The file is scanned
    in chunks so a multi-month log does not have to become one DataFrame
    before the filter.
    """
    csv_path = resolve_data_path(path)
    start_ts = pd.Timestamp(start) if start is not None else None
    end_ts = pd.Timestamp(end) if end is not None else None
    pieces = []
    kept = 0
    for chunk in pd.read_csv(csv_path, chunksize=200_000):
        if "timestamp" not in chunk.columns:
            raise ValueError(f"{csv_path} has no timestamp column. Columns: {list(chunk.columns)}")
        chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], errors="coerce")
        if end_ts is not None and chunk["timestamp"].notna().any() and chunk["timestamp"].min() >= end_ts:
            break
        if start_ts is not None:
            chunk = chunk[chunk["timestamp"] >= start_ts]
        if end_ts is not None:
            chunk = chunk[chunk["timestamp"] < end_ts]
        if chunk.empty:
            continue
        if max_rows is not None:
            room = max_rows - kept
            if room <= 0:
                break
            chunk = chunk.iloc[:room]
        pieces.append(chunk)
        kept += len(chunk)
        if max_rows is not None and kept >= max_rows:
            break
    if not pieces:
        return pd.DataFrame()
    frame = pd.concat(pieces, ignore_index=True)
    return frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


def replay(path: Path | None = None, delay: float = 0.0, start=None, end=None,
           max_rows: int | None = None) -> Iterator[dict]:
    """Yield raw sensor rows in time order, optionally sleeping `delay` seconds."""
    frame = load_window(start=start, end=end, path=path, max_rows=max_rows)
    for record in frame.to_dict(orient="records"):
        yield record
        if delay:
            time.sleep(delay)
