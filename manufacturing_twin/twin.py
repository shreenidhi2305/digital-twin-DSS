"""
twin.py

The Digital Twin itself: a live, in-memory mirror of one physical
machine's state. Each call to update() ingests a new sensor reading,
scores it with the model, and updates the twin's current health.

This is deliberately "lightweight" per the project brief — the twin's
job is to turn raw telemetry into a small, decision-ready evidence
packet for the DSS, not to be a full simulation engine.
"""

from datetime import datetime, timezone

import predictor

# Failure-probability thresholds used to classify machine health.
# These are placeholders — tune them against a validation set / a
# precision-recall curve, since the training data is heavily
# imbalanced (~2% positive class in the sample stream).
HEALTHY_THRESHOLD = 0.30
WARNING_THRESHOLD = 0.60

_RECOMMENDATIONS = {
    "Healthy": "No maintenance required",
    "Warning": "Schedule inspection within the next maintenance window",
    "Critical": "Immediate maintenance required — high failure risk",
}


class MachineTwin:
    """A live digital twin of a single manufacturing machine."""

    def __init__(self, machine_id: str = "CNC-01"):
        self.machine_id = machine_id
        self.latest_state: dict | None = None
        self.history: list[dict] = []  # one entry per update(), for charts/logs

    def update(self, reading: dict) -> dict:
        """Ingest one new sensor reading and refresh the twin's state."""
        result = predictor.predict(reading)
        health = self._classify_health(result["failure_probability"])

        type_code = reading.get("Type")
        type_label = predictor.decode_type(type_code) if type_code is not None else None

        state = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "domain": "Manufacturing",
            "machine": self.machine_id,
            "machine_type": type_label,
            "health": health,
            "failure_probability": result["failure_probability"],
            "air_temperature_K": reading.get("Air temperature [K]", reading.get("Air temperature K")),
            "process_temperature_K": reading.get("Process temperature [K]", reading.get("Process temperature K")),
            "rotational_speed_rpm": reading.get("Rotational speed [rpm]", reading.get("Rotational speed rpm")),
            "torque_Nm": reading.get("Torque [Nm]", reading.get("Torque Nm")),
            "tool_wear_min": reading.get("Tool wear [min]", reading.get("Tool wear min")),
            "recommendation": _RECOMMENDATIONS[health],
        }

        self.latest_state = state
        self.history.append(state)
        return state

    @staticmethod
    def _classify_health(failure_probability: float) -> str:
        if failure_probability < HEALTHY_THRESHOLD:
            return "Healthy"
        elif failure_probability < WARNING_THRESHOLD:
            return "Warning"
        return "Critical"

    def evidence_for_dss(self) -> dict:
        """
        Compact evidence packet for the Decision Support System.
        This — not raw telemetry — is what should cross the boundary
        into the DSS layer of the project.
        """
        if self.latest_state is None:
            return {}
        s = self.latest_state
        return {
            "domain": s["domain"],
            "machine": s["machine"],
            "health": s["health"],
            "failure_probability": s["failure_probability"],
            "tool_wear": s["tool_wear_min"],
            "recommendation": s["recommendation"],
        }


if __name__ == "__main__":
    import stream

    twin = MachineTwin("CNC-01")
    for i, reading in enumerate(stream.replay(delay=0)):
        state = twin.update(reading)
        if i % 400 == 0:
            print(state)
    print("\nFinal DSS evidence packet:")
    print(twin.evidence_for_dss())
