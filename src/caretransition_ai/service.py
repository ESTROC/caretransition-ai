import json
from functools import lru_cache

import joblib
import pandas as pd

from .config import METADATA_PATH, MODEL_PATH

OPTIONAL_FIELDS = {
    "abnormal_lab_rate", "hemoglobin_g_dl", "creatinine_mg_dl",
    "sodium_mmol_l", "glucose_mg_dl", "followup_days"
}


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model artifact missing: {MODEL_PATH}. Run training first.")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def load_metadata():
    if not METADATA_PATH.exists():
        return {"model_version": "unknown", "decision_threshold": 0.5, "model_name": "unknown"}
    return json.loads(METADATA_PATH.read_text())


def risk_band(probability: float) -> str:
    if probability < 0.20:
        return "low"
    if probability < 0.40:
        return "moderate"
    if probability < 0.65:
        return "high"
    return "very_high"


def predict_one(payload: dict) -> dict:
    model = load_model()
    meta = load_metadata()
    threshold = float(meta.get("decision_threshold", 0.5))
    missing = sorted(field for field in OPTIONAL_FIELDS if payload.get(field) is None)

    probability = float(model.predict_proba(pd.DataFrame([payload]))[:, 1][0])
    return {
        "risk_probability": round(probability, 6),
        "prediction": int(probability >= threshold),
        "risk_band": risk_band(probability),
        "threshold": threshold,
        "model_version": str(meta.get("model_version", "unknown")),
        "missing_optional_fields": missing,
    }


def predict_batch(payloads: list[dict]) -> list[dict]:
    return [predict_one(payload) for payload in payloads]
