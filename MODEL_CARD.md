# CareTransition AI - Model Card

## Purpose
Estimate 30-day hospital readmission risk from structured encounter features as an ML engineering demonstration.

## Training data
- Source: synthetic generator included in this repository
- Encounters: 30,000
- Training rows: 19,200
- Validation rows: 4,800
- Held-out test rows: 6,000
- PHI: none

## Model development
Compared:
- Logistic Regression
- Random Forest
- Gradient Boosting

Selection prioritizes validation Average Precision and then ROC-AUC.

## Selected model
Logistic Regression

## Held-out test metrics
- Accuracy: 0.6687
- Precision: 0.4329
- Recall: 0.6701
- F1: 0.5260
- ROC-AUC: 0.7347
- Average Precision: 0.5344
- Brier Score: 0.2044
- Decision threshold: 0.48

## Limitations
The dataset is synthetic. Performance is not evidence of real clinical utility. Real healthcare deployment requires external and temporal validation, calibration review, subgroup/fairness evaluation, privacy/security controls, human oversight, monitoring and regulatory assessment.
