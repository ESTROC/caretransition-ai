from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "synthetic_readmissions.csv"
MODEL_PATH = Path(os.getenv("MODEL_PATH", PROJECT_ROOT / "artifacts" / "readmission_pipeline.joblib"))
METADATA_PATH = Path(os.getenv("METADATA_PATH", PROJECT_ROOT / "artifacts" / "model_metadata.json"))
METRICS_PATH = PROJECT_ROOT / "reports" / "metrics.json"
DRIFT_BASELINE_PATH = PROJECT_ROOT / "artifacts" / "drift_baseline.json"

RANDOM_STATE = 42
DECISION_THRESHOLD = 0.48

NUMERIC_FEATURES = [
    "age", "length_of_stay_days", "prior_admissions_6m", "prior_ed_visits_6m",
    "prior_readmissions_12m", "comorbidity_index", "medication_count",
    "abnormal_lab_rate", "hemoglobin_g_dl", "creatinine_mg_dl",
    "sodium_mmol_l", "glucose_mg_dl", "followup_days", "discharge_readiness_score",
]

CATEGORICAL_FEATURES = ["admission_type", "discharge_disposition", "primary_condition_group"]
RAW_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
