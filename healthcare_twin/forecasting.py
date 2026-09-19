"""
forecasting.py
================
Lightweight AI-assisted admission/demand forecasting with modest
predictive performance. NOT a highly accurate model -- described honestly
throughout as a coarse future-demand signal for the DSS, not a clinical
forecasting tool.

Target: admissions, FORECAST_HORIZON_HOURS ahead (6h).
Model:  Linear Regression (validated during earlier experimentation to be
        at least as good as XGBoost on this data, so the simpler model is
        kept per config/brief).
Validation: time-ordered train/test split (no shuffling -- this is a time
        series, so a random split would leak future information into
        training).

Two variants are trained and reported, matching the earlier validation:
  1. admissions-history only
  2. admissions-history + flu_cases-derived features
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

import config


def _make_features(df: pd.DataFrame, use_flu: bool) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    for lag in config.ADMISSION_LAGS:
        feat[f"admissions_lag{lag}"] = df["admissions"].shift(lag)
    if use_flu:
        for lag in config.FLU_LAGS:
            feat[f"flu_lag{lag}"] = df["flu_cases"].shift(lag)
    feat["target"] = df["admissions"].shift(-config.FORECAST_HORIZON_HOURS)
    return feat.dropna()


def _fit_and_validate(feat: pd.DataFrame):
    X = feat.drop(columns=["target"])
    y = feat["target"]

    split = int(len(feat) * config.TRAIN_FRACTION)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    return model, {"mae": float(mae), "r2": float(r2), "n_test": int(len(y_test))}, \
        (X_test.index, y_test, preds)


def run_forecasting(ops_df: pd.DataFrame) -> dict:
    """
    Returns a dict with both model variants' validation metrics and the
    fitted admissions+flu model's test-set predictions (for plotting).
    """
    feat_admissions_only = _make_features(ops_df, use_flu=False)
    feat_with_flu = _make_features(ops_df, use_flu=True)

    _, metrics_admissions_only, _ = _fit_and_validate(feat_admissions_only)
    model_flu, metrics_with_flu, test_slice = _fit_and_validate(feat_with_flu)

    idx, y_test, preds = test_slice

    return {
        "description": ("Lightweight AI-assisted admission/demand forecasting with "
                         "modest predictive performance. A supplementary future-demand "
                         "signal for the DSS, not a clinical prediction system, and not "
                         "the direct driver of the occupancy simulator (the simulator "
                         "is driven by demand_intensity, see demand_intensity.py)."),
        "horizon_hours": config.FORECAST_HORIZON_HOURS,
        "model": "LinearRegression",
        "validation_method": "time-ordered train/test split (80/20, no shuffling)",
        "admissions_only": metrics_admissions_only,
        "admissions_plus_flu": metrics_with_flu,
        "test_timestamps": ops_df.loc[idx, "timestamp"].tolist(),
        "test_actual": y_test.tolist(),
        "test_predicted": preds.tolist(),
    }
