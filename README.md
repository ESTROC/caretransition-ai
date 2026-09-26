# CareTransition AI

Production-oriented healthcare machine-learning system for **30-day hospital readmission risk estimation** using **Python, Scikit-learn, FastAPI, Docker and CI**.

The repository demonstrates an end-to-end applied ML workflow:

`synthetic cohort generation -> clinical feature engineering -> preprocessing -> model comparison -> held-out evaluation -> API serving -> batch scoring -> drift diagnostics -> Docker -> automated tests`

> **Important:** This project uses synthetic data only. It contains no PHI and is not a medical device or clinical decision-support system.

## Why this project

Readmission risk is a useful healthcare ML problem because it combines:

- numerical and categorical features
- missing clinical values
- utilization history
- class imbalance
- probability estimation
- model evaluation beyond accuracy
- monitoring and regulated-domain engineering concerns

## Input features

Examples include:

- age
- length of stay
- previous admissions and ED visits
- previous readmissions
- comorbidity index
- medication count
- abnormal-lab rate
- hemoglobin
- creatinine
- sodium
- glucose
- follow-up delay
- discharge-readiness score
- admission type
- discharge disposition
- primary condition group

The pipeline also creates features such as utilization burden, transition gap, lab instability and length-of-stay complexity.

## Models compared

- Logistic Regression
- Random Forest
- Gradient Boosting

Model selection prioritizes **Average Precision** and **ROC-AUC**, which are more informative than raw accuracy for an imbalanced risk-prediction problem.

## Verified held-out result

The reproducible run uses **30,000 synthetic encounters**, split into **19,200 training**, **4,800 validation**, and **6,000 held-out test** rows.

Selected model: **Logistic Regression**

| Metric | Test result |
| --- | ---: |
| Accuracy | 66.87% |
| Precision | 43.29% |
| Recall | 67.01% |
| F1 | 52.60% |
| ROC-AUC | 0.7347 |
| Average Precision | 0.5344 |
| Brier Score | 0.2044 |
| Decision threshold | 0.48 |

These metrics demonstrate the engineering workflow on synthetic data, not clinical effectiveness.

## API

- `GET /health`
- `GET /metadata`
- `POST /predict`
- `POST /predict/batch`
- `POST /monitor/drift`
- Swagger/OpenAPI at `/docs`

## Quick start

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
# source .venv/bin/activate

pip install -e ".[dev]"
python -m caretransition_ai.data_generation
python -m caretransition_ai.train
pytest -q
uvicorn caretransition_ai.api:app --reload
```

Open `http://127.0.0.1:8000/docs`.

## Docker

```bash
docker compose up --build
```

The multi-stage Docker build regenerates the synthetic cohort and trains the model, keeping generated patient-like data and model binaries out of Git.

## Production-oriented engineering

- one serialized preprocessing + model pipeline
- train/validation/untouched-test separation
- missing-value handling
- one-hot encoding
- probability-based inference
- validation-selected classification threshold
- batch scoring
- model metadata
- lightweight drift diagnostics
- non-root Docker runtime
- container health check
- automated tests
- GitHub Actions CI
- model card and security guidance

## Safety and limitations

CareTransition AI is a portfolio and ML engineering demonstration. A real healthcare deployment would require external and temporal validation, probability calibration, subgroup/fairness evaluation, PHI/privacy controls, authentication, authorization, audit logging, encrypted storage/transport, monitoring, model governance, clinician oversight, and regulatory assessment.

## License

MIT License.