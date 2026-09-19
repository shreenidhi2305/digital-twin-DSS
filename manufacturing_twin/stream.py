"""
stream.py

Simulates a live sensor feed by replaying live_machine_stream.csv
row by row. In a real deployment this module is the seam you'd swap
out for a Kafka/MQTT consumer or a live PLC/OPC-UA connection —
everything downstream (twin.py, dashboard.py) only depends on getting
a dict per row, so the rest of the pipeline doesn't change.
"""

import time
from pathlib import Path
from typing import Iterator

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "data" / "live_machine_stream.csv"


def load_stream(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the full stream as a DataFrame (useful for batch/offline use)."""
    return pd.read_csv(path)


def replay(path: Path = DATA_PATH, delay: float = 0.0) -> Iterator[dict]:
    """
    Generator that yields one sensor reading (as a dict) at a time, in
    row order, optionally sleeping `delay` seconds between rows to
    simulate a real-time feed.
    """
    df = load_stream(path)
    for _, row in df.iterrows():
        yield row.to_dict()
        if delay:
            time.sleep(delay)


if __name__ == "__main__":
    for i, reading in enumerate(replay(delay=0)):
        print(reading)
        if i >= 4:
            break
