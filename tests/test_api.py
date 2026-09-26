from fastapi.testclient import TestClient

from caretransition_ai.api import app

PATIENT = {
    "age": 72, "length_of_stay_days": 5.2, "prior_admissions_6m": 1,
    "prior_ed_visits_6m": 2, "prior_readmissions_12m": 1,
    "comorbidity_index": 4, "medication_count": 12,
    "abnormal_lab_rate": 0.28, "hemoglobin_g_dl": 11.8,
    "creatinine_mg_dl": 1.5, "sodium_mmol_l": 136,
    "glucose_mg_dl": 158, "followup_days": 10,
    "discharge_readiness_score": 68, "admission_type": "emergency",
    "discharge_disposition": "home_health",
    "primary_condition_group": "cardiovascular",
}


def test_health_prediction_batch_and_drift():
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/metadata").status_code == 200

        prediction = client.post("/predict", json=PATIENT)
        assert prediction.status_code == 200
        assert 0 <= prediction.json()["risk_probability"] <= 1

        batch = client.post("/predict/batch", json={"patients": [PATIENT, {**PATIENT, "age": 82}]})
        assert batch.status_code == 200
        assert batch.json()["n"] == 2

        drift = client.post("/monitor/drift", json={"patients": [PATIENT, {**PATIENT, "age": 82}]})
        assert drift.status_code == 200
        assert drift.json()["available"] is True

        invalid = client.post("/predict", json={**PATIENT, "age": 250})
        assert invalid.status_code == 422
