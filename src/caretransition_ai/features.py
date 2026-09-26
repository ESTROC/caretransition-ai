import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Deterministic healthcare-domain feature engineering inside the ML pipeline."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        df["utilization_burden"] = (
            df["prior_admissions_6m"] + df["prior_ed_visits_6m"]
            + 2.0 * df["prior_readmissions_12m"]
        )
        df["meds_per_comorbidity"] = df["medication_count"] / (df["comorbidity_index"] + 1.0)
        df["renal_stress"] = df["creatinine_mg_dl"] * (1.0 + df["abnormal_lab_rate"].fillna(0))
        df["lab_instability"] = (
            np.abs(df["sodium_mmol_l"] - 139.0) / 10.0
            + np.maximum(12.0 - df["hemoglobin_g_dl"], 0).fillna(0) / 5.0
            + np.maximum(df["glucose_mg_dl"] - 140.0, 0).fillna(0) / 150.0
        )
        df["transition_gap"] = df["followup_days"] / 7.0 + (100.0 - df["discharge_readiness_score"]) / 30.0
        df["los_complexity"] = df["length_of_stay_days"] * (1.0 + df["comorbidity_index"] / 10.0)
        return df
