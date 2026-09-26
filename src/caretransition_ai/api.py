from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

from .drift import numeric_drift_snapshot
from .schemas import BatchRequest, BatchResponse, PatientFeatures, PredictionResponse
from .service import load_metadata, load_model, predict_batch, predict_one


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    load_metadata()
    yield


app = FastAPI(
    title="CareTransition AI",
    version="1.0.0",
    description="30-day hospital readmission risk ML demonstration using synthetic data.",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "service": "CareTransition AI",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "disclaimer": "Demonstration only; not a medical device.",
    }


@app.get("/health")
def health():
    meta = load_metadata()
    return {
        "status": "ok",
        "model_name": meta.get("model_name", "unknown"),
        "model_version": meta.get("model_version", "unknown"),
    }


@app.get("/metadata")
def metadata():
    meta = load_metadata()
    return {
        "project": meta.get("project"),
        "model_name": meta.get("model_name"),
        "model_version": meta.get("model_version"),
        "decision_threshold": meta.get("decision_threshold"),
        "raw_features": meta.get("raw_features"),
        "test_metrics": meta.get("test_metrics"),
        "top_features": meta.get("top_features"),
        "synthetic_training_data": meta.get("synthetic_training_data"),
        "intended_use": meta.get("intended_use"),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PatientFeatures):
    try:
        return predict_one(payload.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Inference failed") from exc


@app.post("/predict/batch", response_model=BatchResponse)
def batch(payload: BatchRequest):
    try:
        rows = [patient.model_dump() for patient in payload.patients]
        return {"n": len(rows), "results": predict_batch(rows)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Batch inference failed") from exc


@app.post("/monitor/drift")
def drift(payload: BatchRequest):
    rows = [patient.model_dump() for patient in payload.patients]
    return numeric_drift_snapshot(pd.DataFrame(rows))
