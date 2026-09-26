import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    CATEGORICAL_FEATURES, DATA_PATH, DRIFT_BASELINE_PATH, METADATA_PATH,
    METRICS_PATH, MODEL_PATH, NUMERIC_FEATURES, RANDOM_STATE, RAW_FEATURES
)
from .data_generation import generate_dataset
from .features import ClinicalFeatureEngineer

ENGINEERED_NUMERIC = [
    "utilization_burden", "meds_per_comorbidity", "renal_stress",
    "lab_instability", "transition_gap", "los_complexity"
]


def build_pipeline(model):
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES + ENGINEERED_NUMERIC),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ], remainder="drop", verbose_feature_names_out=False)

    return Pipeline([
        ("feature_engineering", ClinicalFeatureEngineer()),
        ("preprocess", preprocessor),
        ("model", model),
    ])


def score(y_true, prob, threshold):
    pred = (prob >= threshold).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, prob)), 4),
        "average_precision": round(float(average_precision_score(y_true, prob)), 4),
        "brier_score": round(float(brier_score_loss(y_true, prob)), 4),
        "confusion_matrix": confusion_matrix(y_true, pred).tolist(),
        "threshold": threshold,
    }


def build_drift_baseline(frame):
    baseline = {"numeric": {}, "categorical": {}}
    for col in NUMERIC_FEATURES:
        s = pd.to_numeric(frame[col], errors="coerce").dropna()
        baseline["numeric"][col] = {
            "mean": round(float(s.mean()), 6),
            "std": round(float(s.std(ddof=0)), 6),
            "p05": round(float(s.quantile(0.05)), 6),
            "p50": round(float(s.quantile(0.50)), 6),
            "p95": round(float(s.quantile(0.95)), 6),
        }
    for col in CATEGORICAL_FEATURES:
        freq = frame[col].fillna("UNKNOWN").value_counts(normalize=True)
        baseline["categorical"][col] = {str(k): round(float(v), 6) for k, v in freq.items()}
    return baseline


def train():
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        generate_dataset().to_csv(DATA_PATH, index=False)

    df = pd.read_csv(DATA_PATH)
    X = df[RAW_FEATURES]
    y = df["readmitted_30d"].astype(int)

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.20, stratify=y_trainval, random_state=RANDOM_STATE
    )

    candidates = {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=350, max_depth=12, min_samples_leaf=5,
            class_weight="balanced_subsample", n_jobs=-1, random_state=RANDOM_STATE
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=180, learning_rate=0.04, max_depth=3, random_state=RANDOM_STATE
        ),
    }

    validation = {}
    thresholds = {}
    for name, estimator in candidates.items():
        pipe = build_pipeline(estimator)
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_val)[:, 1]

        threshold_grid = np.linspace(0.20, 0.80, 61)
        best_threshold = max(
            ((float(t), f1_score(y_val, (prob >= t).astype(int), zero_division=0)) for t in threshold_grid),
            key=lambda item: item[1]
        )[0]

        thresholds[name] = best_threshold
        validation[name] = score(y_val, prob, best_threshold)

    best_name = max(
        validation,
        key=lambda n: (validation[n]["average_precision"], validation[n]["roc_auc"])
    )
    selected_threshold = float(thresholds[best_name])

    final_model = build_pipeline(candidates[best_name])
    final_model.fit(X_trainval, y_trainval)

    test_prob = final_model.predict_proba(X_test)[:, 1]
    test_metrics = score(y_test, test_prob, selected_threshold)

    importance_result = permutation_importance(
        final_model, X_test, y_test, scoring="roc_auc",
        n_repeats=5, random_state=RANDOM_STATE, n_jobs=-1
    )
    importance = sorted([
        {
            "feature": feature,
            "importance_mean": round(float(mean), 6),
            "importance_std": round(float(std), 6),
        }
        for feature, mean, std in zip(
            RAW_FEATURES, importance_result.importances_mean, importance_result.importances_std
        )
    ], key=lambda row: row["importance_mean"], reverse=True)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, MODEL_PATH)

    metrics = {
        "dataset": {
            "rows": int(len(df)),
            "readmission_rate": round(float(y.mean()), 4),
            "train_rows": int(len(X_train)),
            "validation_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
            "synthetic": True,
        },
        "validation_model_comparison": validation,
        "selected_model": best_name,
        "test_metrics": test_metrics,
        "permutation_importance": importance,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    metadata = {
        "project": "CareTransition AI",
        "model_name": best_name,
        "model_version": "1.0.0",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision_threshold": selected_threshold,
        "raw_features": RAW_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "synthetic_training_data": True,
        "intended_use": "ML engineering demonstration only; not a medical device",
        "test_metrics": test_metrics,
        "top_features": importance[:8],
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    DRIFT_BASELINE_PATH.write_text(json.dumps(build_drift_baseline(X_trainval), indent=2))

    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train()
