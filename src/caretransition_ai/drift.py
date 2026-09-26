import json
import pandas as pd

from .config import DRIFT_BASELINE_PATH, NUMERIC_FEATURES


def numeric_drift_snapshot(frame: pd.DataFrame) -> dict:
    """Lightweight standardized mean-shift diagnostics against the training baseline."""
    if not DRIFT_BASELINE_PATH.exists():
        return {"available": False, "features": {}}

    baseline = json.loads(DRIFT_BASELINE_PATH.read_text())["numeric"]
    output = {}

    for col in NUMERIC_FEATURES:
        if col not in frame:
            continue
        series = pd.to_numeric(frame[col], errors="coerce").dropna()
        if series.empty:
            continue

        b = baseline[col]
        denom = max(abs(float(b["std"])), 1e-8)
        z_shift = abs(float(series.mean()) - float(b["mean"])) / denom
        output[col] = {
            "current_mean": round(float(series.mean()), 6),
            "training_mean": b["mean"],
            "standardized_mean_shift": round(z_shift, 4),
            "flagged": bool(z_shift >= 1.0),
        }

    return {
        "available": True,
        "features": output,
        "flagged_features": [name for name, values in output.items() if values["flagged"]],
    }
