"""
check_environment.py

Run this FIRST, before `streamlit run dashboard.py`, whenever you set up
a new venv or move the project to a new machine. It isolates the model
loading step from everything else, so a version mismatch or a bad file
shows up immediately with a clear message instead of buried in a
Streamlit traceback.

Usage:
    python check_environment.py
"""

import sys
from pathlib import Path

print("Python:", sys.version)

try:
    import xgboost
    print("xgboost:", xgboost.__version__, "(trained with 3.3.0, but any recent version reads the .json model fine)")
except ImportError:
    print("xgboost is NOT installed. Run: pip install -r requirements.txt")
    sys.exit(1)

try:
    import sklearn
    print("scikit-learn:", sklearn.__version__, "(model was trained with 1.6.1)")
except ImportError:
    print("scikit-learn is NOT installed.")
    sys.exit(1)

import joblib
print("joblib:", joblib.__version__)

MODEL_DIR = Path(__file__).resolve().parent / "models"

# machine_failure_xgboost.json is the file the app actually loads (via
# XGBoost's own portable format). The .pkl is only kept around as an
# archival/backward-compat artifact from the training notebook, so its
# absence is a warning, not a fatal error.
print("\nChecking file sizes...")
for fname in ["machine_failure_xgboost.json", "label_encoder.pkl", "feature_columns.pkl"]:
    path = MODEL_DIR / fname
    if not path.exists():
        print(f"  MISSING: {fname}")
        continue
    print(f"  {fname}: {path.stat().st_size} bytes")

pkl_path = MODEL_DIR / "machine_failure_xgboost.pkl"
if pkl_path.exists():
    print(f"  machine_failure_xgboost.pkl: {pkl_path.stat().st_size} bytes (unused by the app, kept for archival)")
else:
    print("  machine_failure_xgboost.pkl: not present (fine — the app doesn't need it)")

print("\nAttempting to load each artifact...")
try:
    fc = joblib.load(MODEL_DIR / "feature_columns.pkl")
    print("  feature_columns.pkl -> OK:", fc)
except Exception as e:
    print("  feature_columns.pkl -> FAILED:", type(e).__name__, e)

try:
    le = joblib.load(MODEL_DIR / "label_encoder.pkl")
    print("  label_encoder.pkl -> OK, classes:", list(le.classes_))
except Exception as e:
    print("  label_encoder.pkl -> FAILED:", type(e).__name__, e)

try:
    from xgboost import XGBClassifier
    model = XGBClassifier()
    model.load_model(MODEL_DIR / "machine_failure_xgboost.json")
    print("  machine_failure_xgboost.json -> OK:", type(model))
except Exception as e:
    print("  machine_failure_xgboost.json -> FAILED:", type(e).__name__, e)
    print("\n  This format doesn't carry a version tag the way the old .pkl did,")
    print("  so a failure here is more likely a missing/corrupted file than a")
    print("  version mismatch. Re-export it from the training notebook with")
    print("  best_model.save_model('machine_failure_xgboost.json') and re-copy")
    print("  it into the models/ folder.")
    sys.exit(1)

print("\nRunning a test prediction...")
import pandas as pd
sample = {
    "Type": 1,
    "Air temperature K": 298.4,
    "Process temperature K": 308.7,
    "Rotational speed rpm": 1500,
    "Torque Nm": 45.0,
    "Tool wear min": 120,
}
X = pd.DataFrame([sample])[fc]
proba = model.predict_proba(X)[0, 1]
print(f"  Test prediction succeeded. Failure probability: {proba:.4f}")
print("\nEverything checks out — safe to run: streamlit run dashboard.py")
