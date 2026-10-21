"""
Check that the MetroPT-3 SAE artifacts load and the checksum manifest matches.

Usage (from manufacturing_twin/):
    python check_environment.py
"""

import sys
from pathlib import Path

print("Python:", sys.version)

try:
    import torch
    print("torch:", torch.__version__)
except ImportError:
    print("torch is NOT installed. From the repo root:")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cpu")
    sys.exit(1)

try:
    import numpy as np
    import pandas as pd
    print("pandas:", pd.__version__)
    print("numpy:", np.__version__)
except ImportError as exc:
    print("pandas/numpy missing:", exc)
    sys.exit(1)

TWIN = Path(__file__).resolve().parent
if str(TWIN) not in sys.path:
    sys.path.insert(0, str(TWIN))

from dt.models.metropt3_sae import PRIMARY_BUDGET, MetroSAEDetector

print("\nLoading SAE (checksums must match MANIFEST.sha256)...")
try:
    detector = MetroSAEDetector.load()
except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
    print("FAILED:", type(exc).__name__, exc)
    sys.exit(1)

print("  features:", len(detector.feature_names))
print("  primary budget", PRIMARY_BUDGET, "threshold", detector.threshold(PRIMARY_BUDGET))
print("  device: cpu, eval mode")
sample = pd.DataFrame([detector.mean], columns=detector.feature_names)
score = float(detector.score(sample)[0])
print(f"  score at the training mean vector: {score:.6g}")
print("\nArtifacts loaded. Replay still needs MetroPT-3 CSV in data/.")
print("Dashboard: streamlit run dashboard_standalone.py")
