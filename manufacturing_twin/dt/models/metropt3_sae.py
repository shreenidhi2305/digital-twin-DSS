"""
Sparse autoencoder anomaly detector for the MetroPT-3 air-compressor (APU).

Inference only. Training stays in the notebook that produced the artifacts.
The score is the per-row reconstruction MSE on the 23 standardised features.
It is an anomaly score, not a failure probability and not a remaining useful life.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from dt.features.metropt3 import FEATURE_NAMES

# Files copied from the training run. MANIFEST.sha256 lists their SHA-256.
REQUIRED_ARTIFACTS = (
    "sae_model_state.pt",
    "scaler.json",
    "feature_names.json",
    "thresholds.json",
    "run_config.json",
    "training_history.csv",
    "test_event_results.csv",
    "test_false_alarm_summary.csv",
    "test_alarm_episodes.csv",
    "drift_check_train_vs_val.csv",
    "score_drift_monthly.csv",
)

DEFAULT_MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "metropt3_sae"
PRIMARY_BUDGET = "1.0"


class MetroSAE(nn.Module):
    """Linear(23, 64) -> ReLU -> Linear(64, 23). Keys match sae_model_state.pt."""

    def __init__(self, n_features: int = 23, hidden_dim: int = 64):
        super().__init__()
        self.enc = nn.Linear(n_features, hidden_dim)
        self.dec = nn.Linear(hidden_dim, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dec(torch.relu(self.enc(x)))


class ArtifactChecksumError(RuntimeError):
    """Raised when an artifact's bytes do not match MANIFEST.sha256."""


def verify_manifest(model_dir: Path) -> dict[str, str]:
    """
    Recompute SHA-256 for every file named in MANIFEST.sha256 and refuse to
    continue if any digest differs or a required artifact is absent.
    """
    model_dir = Path(model_dir)
    manifest_path = model_dir / "MANIFEST.sha256"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing {manifest_path}. Refusing to load the SAE without a checksum manifest."
        )

    expected: dict[str, str] = {}
    for line_no, raw in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            raise ArtifactChecksumError(f"{manifest_path}:{line_no} is not 'sha256  filename'")
        digest, name = parts[0].lower(), parts[-1].replace("\\", "/")
        if name in {"MANIFEST.sha256", ".", ".."} or "/" in name or name.startswith("."):
            raise ArtifactChecksumError(f"Manifest entry {name!r} is not a file in {model_dir}")
        expected[name] = digest

    missing = [name for name in REQUIRED_ARTIFACTS if name not in expected]
    if missing:
        raise ArtifactChecksumError(f"MANIFEST.sha256 does not list required artifacts: {missing}")

    for name, digest in expected.items():
        path = model_dir / name
        if not path.is_file():
            raise ArtifactChecksumError(f"Manifest lists {name} but the file is missing")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != digest:
            raise ArtifactChecksumError(
                f"Checksum mismatch for {name}: manifest {digest}, file {actual}. "
                "The artifact was edited or copied incompletely. Refusing to score."
            )
    return expected


def _budget_key(budget: str | float) -> str:
    if isinstance(budget, str):
        key = budget.strip()
        try:
            key = f"{float(key):.1f}"
        except ValueError:
            pass
    else:
        key = f"{float(budget):.1f}"
    return key


class MetroSAEDetector:
    """Frozen scaler, frozen thresholds, CPU inference."""

    def __init__(self, model: MetroSAE, feature_names: list[str], mean: np.ndarray,
                 std: np.ndarray, thresholds: dict, merge_minutes: float, model_dir: Path):
        if list(feature_names) != list(FEATURE_NAMES):
            raise ValueError(
                "feature_names.json does not match the feature pipeline order.\n"
                f"  file: {feature_names}\n  pipeline: {list(FEATURE_NAMES)}"
            )
        if mean.shape != (len(FEATURE_NAMES),) or std.shape != (len(FEATURE_NAMES),):
            raise ValueError(
                f"Scaler length {mean.shape}/{std.shape} != {len(FEATURE_NAMES)} features"
            )
        if np.any(std <= 0):
            raise ValueError("Scaler has a non-positive standard deviation")
        self.model = model
        self.feature_names = list(feature_names)
        self.mean = mean.astype(np.float64)
        self.std = std.astype(np.float64)
        self.thresholds = thresholds
        self.merge_minutes = float(merge_minutes)
        self.model_dir = Path(model_dir)
        self.primary_budget = PRIMARY_BUDGET

    @classmethod
    def load(cls, model_dir: Path | None = None) -> MetroSAEDetector:
        model_dir = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
        verify_manifest(model_dir)

        feature_names = json.loads((model_dir / "feature_names.json").read_text(encoding="utf-8"))
        scaler = json.loads((model_dir / "scaler.json").read_text(encoding="utf-8"))
        thresholds_doc = json.loads((model_dir / "thresholds.json").read_text(encoding="utf-8"))
        run_config = json.loads((model_dir / "run_config.json").read_text(encoding="utf-8"))

        if list(scaler.get("feature_names", [])) != list(feature_names):
            raise ValueError("scaler.json feature_names differ from feature_names.json")

        state = torch.load(model_dir / "sae_model_state.pt", map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise TypeError(f"Expected a state dict, got {type(state)}")
        # The checkpoint uses enc.* and dec.*. Build the module from those
        # tensor shapes so a width change fails here instead of silently slicing.
        required = {"enc.weight", "enc.bias", "dec.weight", "dec.bias"}
        if set(state) != required:
            raise KeyError(
                "sae_model_state.pt keys do not match MetroSAE (enc/dec). "
                f"Found {sorted(state)}. Refusing to rename weights."
            )
        enc_w = state["enc.weight"]
        dec_w = state["dec.weight"]
        n_in, hidden = int(enc_w.shape[1]), int(enc_w.shape[0])
        if dec_w.shape != (n_in, hidden):
            raise ValueError(f"Decoder shape {tuple(dec_w.shape)} is not ({n_in}, {hidden})")
        if n_in != len(feature_names):
            raise ValueError(
                f"Encoder input width {n_in} != {len(feature_names)} feature names"
            )

        model = MetroSAE(n_features=n_in, hidden_dim=hidden)
        model.load_state_dict(state, strict=True)
        model.to("cpu")
        model.eval()

        thresholds = {}
        for key, payload in thresholds_doc["thresholds"].items():
            thresholds[_budget_key(key)] = float(payload["threshold"])
        if PRIMARY_BUDGET not in thresholds:
            raise KeyError(f"thresholds.json has no primary budget {PRIMARY_BUDGET}")

        merge_minutes = float(run_config["config"]["merge_minutes"])
        return cls(
            model=model,
            feature_names=list(feature_names),
            mean=np.asarray(scaler["mean"], dtype=np.float64),
            std=np.asarray(scaler["std"], dtype=np.float64),
            thresholds=thresholds,
            merge_minutes=merge_minutes,
            model_dir=model_dir,
        )

    def threshold(self, budget: str | float = PRIMARY_BUDGET) -> float:
        key = _budget_key(budget)
        if key not in self.thresholds:
            known = ", ".join(sorted(self.thresholds))
            raise KeyError(f"Unknown alarm budget {budget!r}. Frozen budgets: {known}")
        return self.thresholds[key]

    def _matrix(self, df_features) -> np.ndarray:
        columns = list(df_features.columns)
        if columns != self.feature_names:
            raise ValueError(
                "Feature frame columns must equal feature_names.json in the same order. "
                f"Got {columns}"
            )
        values = np.asarray(df_features.to_numpy(), dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError(f"Expected shape (n, {len(self.feature_names)}), got {values.shape}")
        return values

    def _standardised(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (z, finite_row_mask). Non-finite rows stay NaN. No imputation."""
        finite = np.isfinite(values).all(axis=1)
        z = np.full(values.shape, np.nan, dtype=np.float64)
        if finite.any():
            z[finite] = (values[finite] - self.mean) / self.std
        return z, finite

    def _reconstruct(self, z: np.ndarray, finite: np.ndarray) -> np.ndarray:
        recon = np.full(z.shape, np.nan, dtype=np.float64)
        if not finite.any():
            return recon
        block = torch.from_numpy(np.ascontiguousarray(z[finite], dtype=np.float32))
        self.model.eval()
        with torch.no_grad():
            out = self.model(block).detach().cpu().numpy().astype(np.float64)
        recon[finite] = out
        return recon

    def score(self, df_features) -> np.ndarray:
        """Per-row MSE of the reconstruction. NaN where any input feature is non-finite."""
        values = self._matrix(df_features)
        z, finite = self._standardised(values)
        recon = self._reconstruct(z, finite)
        err = z - recon
        scores = np.full(len(values), np.nan, dtype=np.float64)
        if finite.any():
            scores[finite] = np.mean(err[finite] ** 2, axis=1)
        return scores

    def feature_contributions(self, df_features) -> np.ndarray:
        """Per-feature squared error on the standardised features. Shape (n, 23)."""
        values = self._matrix(df_features)
        z, finite = self._standardised(values)
        recon = self._reconstruct(z, finite)
        contrib = np.full(z.shape, np.nan, dtype=np.float64)
        if finite.any():
            contrib[finite] = (z[finite] - recon[finite]) ** 2
        return contrib

    def is_alarm(self, score, budget: str | float = PRIMARY_BUDGET):
        """
        True where a finite score is at or above the frozen threshold for `budget`.
        Non-finite scores are not alarms. A scalar score returns a bool.
        """
        threshold = self.threshold(budget)
        arr = np.asarray(score, dtype=np.float64)
        flagged = np.isfinite(arr) & (arr >= threshold)
        if arr.ndim == 0:
            return bool(flagged)
        return flagged
