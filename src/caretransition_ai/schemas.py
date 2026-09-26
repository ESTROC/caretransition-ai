from typing import Literal
from pydantic import BaseModel, Field


class PatientFeatures(BaseModel):
    age: float = Field(..., ge=18, le=110)
    length_of_stay_days: float = Field(..., ge=0, le=365)
    prior_admissions_6m: int = Field(..., ge=0, le=50)
    prior_ed_visits_6m: int = Field(..., ge=0, le=50)
    prior_readmissions_12m: int = Field(..., ge=0, le=50)
    comorbidity_index: int = Field(..., ge=0, le=30)
    medication_count: float = Field(..., ge=0, le=100)
    abnormal_lab_rate: float | None = Field(None, ge=0, le=1)
    hemoglobin_g_dl: float | None = Field(None, ge=3, le=25)
    creatinine_mg_dl: float | None = Field(None, ge=0.1, le=20)
    sodium_mmol_l: float | None = Field(None, ge=100, le=180)
    glucose_mg_dl: float | None = Field(None, ge=20, le=700)
    followup_days: float | None = Field(None, ge=0, le=90)
    discharge_readiness_score: float = Field(..., ge=0, le=100)
    admission_type: Literal["emergency", "urgent", "elective"]
    discharge_disposition: Literal["home", "home_health", "skilled_nursing", "rehab"]
    primary_condition_group: Literal["cardiovascular", "respiratory", "endocrine", "renal", "other"]


class PredictionResponse(BaseModel):
    risk_probability: float
    prediction: int
    risk_band: str
    threshold: float
    model_version: str
    missing_optional_fields: list[str]


class BatchRequest(BaseModel):
    patients: list[PatientFeatures] = Field(..., min_length=1, max_length=500)


class BatchResponse(BaseModel):
    n: int
    results: list[PredictionResponse]
