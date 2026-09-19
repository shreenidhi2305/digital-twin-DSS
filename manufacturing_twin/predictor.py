"""
predictor.py

Loads the trained XGBoost failure model, the label encoder for the
'Type' feature, and the exact feature schema the model was trained on.
Exposes a single `predict()` function that turns a raw sensor reading
into a failure probability + predicted class.

This module is intentionally the ONLY place that knows about the
model's internals (feature names, encoding, column order). twin.py
and dashboard.py just call predict() and don't need to know how the
model works under the hood.
"""

from pathlib import Path
import joblib
import pandas as pd
from xgboost import XGBClassifier

MODEL_DIR = Path(__file__).resolve().parent / "models"

# --- Load artifacts once, at import time ------------------------------
#
# The model is loaded from machine_failure_xgboost.json, XGBoost's own
# portable serialization format, rather than the .pkl. Pickling a
# booster ties the file to the exact xgboost version that wrote it --
# loading it with any other version raises
# "XGBoostError: input stream corrupted". The .json format is stable
# across xgboost versions, so this bug can't happen again regardless
# of what's installed in whatever venv runs the dashboard.
#
# (machine_failure_xgboost.pkl is still produced by the training
# notebook for backward compatibility / archival, but this app no
# longer reads it.)
_model = XGBClassifier()
_model.load_model(MODEL_DIR / "machine_failure_xgboost.json")

_label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
_feature_columns = joblib.load(MODEL_DIR / "feature_columns.pkl")

# The live CSV uses bracketed units ("Air temperature [K]") but the
# model was trained on column names without brackets ("Air temperature K").
# This maps raw/CSV-style names -> the exact names the model expects.
_COLUMN_RENAME_MAP = {
    "Air temperature [K]": "Air temperature K",
    "Process temperature [K]": "Process temperature K",
    "Rotational speed [rpm]": "Rotational speed rpm",
    "Torque [Nm]": "Torque Nm",
    "Tool wear [min]": "Tool wear min",
}


def decode_type(type_code) -> str:
    """Turn the encoded Type (0/1/2) back into its L/M/H label."""
    return _label_encoder.inverse_transform([int(type_code)])[0]


def prepare_features(reading: dict) -> pd.DataFrame:
    """
    Take a raw sensor reading (dict — may use either the bracketed CSV
    column names or the model's plain names) and return a single-row
    DataFrame with columns in the exact name/order the model expects.
    """
    row = dict(reading)

    for raw_name, model_name in _COLUMN_RENAME_MAP.items():
        if raw_name in row and model_name not in row:
            row[model_name] = row.pop(raw_name)

    # Never let the ground-truth label leak in as a feature, if present.
    row.pop("Machine failure", None)

    df = pd.DataFrame([row])
    missing = [c for c in _feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    return df[_feature_columns]


def predict(reading: dict) -> dict:
    """
    Score a single sensor reading.

    Returns:
        {
            "failure_probability": float in [0, 1],
            "predicted_failure": bool,
        }
    """
    X = prepare_features(reading)
    proba = float(_model.predict_proba(X)[0, 1])
    pred_class = int(_model.predict(X)[0])
    return {
        "failure_probability": round(proba, 4),
        "predicted_failure": bool(pred_class),
    }


if __name__ == "__main__":
    # Quick smoke test
    sample = {
        "Type": 1,
        "Air temperature [K]": 298.4,
        "Process temperature [K]": 308.7,
        "Rotational speed [rpm]": 1500,
        "Torque [Nm]": 45.0,
        "Tool wear [min]": 120,
    }
    print("Type decoded:", decode_type(sample["Type"]))
    print("Prediction:", predict(sample))
