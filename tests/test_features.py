import pandas as pd

from caretransition_ai.features import ClinicalFeatureEngineer


def test_feature_engineering_adds_expected_columns():
    frame = pd.DataFrame([{
        "age": 70, "length_of_stay_days": 5, "prior_admissions_6m": 2,
        "prior_ed_visits_6m": 1, "prior_readmissions_12m": 1,
        "comorbidity_index": 4, "medication_count": 10,
        "abnormal_lab_rate": 0.3, "hemoglobin_g_dl": 11.5,
        "creatinine_mg_dl": 1.4, "sodium_mmol_l": 136,
        "glucose_mg_dl": 160, "followup_days": 10,
        "discharge_readiness_score": 65, "admission_type": "emergency",
        "discharge_disposition": "home_health",
        "primary_condition_group": "cardiovascular",
    }])
    out = ClinicalFeatureEngineer().fit_transform(frame)
    assert "utilization_burden" in out.columns
    assert "transition_gap" in out.columns
    assert "lab_instability" in out.columns
    assert out.loc[0, "utilization_burden"] == 5
