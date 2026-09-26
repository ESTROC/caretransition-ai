import numpy as np
import pandas as pd

from .config import DATA_PATH, RANDOM_STATE


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def generate_dataset(n_samples: int = 30000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """Create a reproducible synthetic cohort for ML engineering demonstrations."""
    rng = np.random.default_rng(seed)

    age = np.clip(rng.normal(66, 15, n_samples), 18, 95)
    admission_type = rng.choice(["emergency", "urgent", "elective"], n_samples, p=[0.55, 0.22, 0.23])
    condition = rng.choice(
        ["cardiovascular", "respiratory", "endocrine", "renal", "other"],
        n_samples, p=[0.27, 0.21, 0.18, 0.14, 0.20]
    )
    discharge = rng.choice(
        ["home", "home_health", "skilled_nursing", "rehab"],
        n_samples, p=[0.57, 0.18, 0.17, 0.08]
    )

    comorbidity = np.clip(
        rng.poisson(2.4, n_samples)
        + (age > 75).astype(int)
        + (condition == "renal").astype(int),
        0, 12
    )
    prior_adm = np.clip(rng.poisson(0.7 + 0.18 * comorbidity), 0, 10)
    prior_ed = np.clip(rng.poisson(0.8 + 0.15 * comorbidity), 0, 12)
    prior_readm = np.clip(rng.poisson(0.20 + 0.13 * prior_adm + 0.08 * prior_ed), 0, 8)

    los = np.clip(
        rng.gamma(2.0, 2.0, n_samples)
        + 0.33 * comorbidity
        + 1.0 * (admission_type == "emergency"),
        0.5, 30
    )

    meds = np.clip(rng.normal(6 + 1.3 * comorbidity + age / 22, 3.0, n_samples), 0, 35)
    abnormal_lab_rate = np.clip(
        rng.beta(2.2, 7.0, n_samples)
        + 0.025 * comorbidity
        + 0.04 * (condition == "renal"),
        0, 1
    )

    hemoglobin = np.clip(
        rng.normal(13.2 - 0.18 * comorbidity - 0.6 * (condition == "renal"), 1.4),
        7.0, 18.0
    )
    creatinine = np.clip(
        rng.lognormal(
            mean=np.log(0.95 + 0.16 * comorbidity + 0.75 * (condition == "renal")),
            sigma=0.30
        ),
        0.3, 8.0
    )
    sodium = np.clip(rng.normal(139 - 0.35 * comorbidity, 3.8, n_samples), 120, 152)
    glucose = np.clip(
        rng.normal(105 + 16 * (condition == "endocrine") + 2.2 * comorbidity, 28),
        55, 350
    )

    followup = np.clip(
        rng.normal(
            9 + 4 * (discharge == "home")
            - 2 * (discharge == "home_health")
            - 2 * prior_readm,
            5
        ),
        1, 30
    )

    readiness = np.clip(
        rng.normal(
            82 - 3.1 * comorbidity - 2.2 * prior_readm
            - 8 * abnormal_lab_rate
            + 4 * (discharge == "home"),
            8
        ),
        20, 100
    )

    logit = (
        -4.00
        + 0.012 * (age - 60)
        + 0.24 * prior_adm
        + 0.28 * prior_ed
        + 0.55 * prior_readm
        + 0.15 * comorbidity
        + 0.023 * meds
        + 1.25 * abnormal_lab_rate
        + 0.11 * np.maximum(creatinine - 1.1, 0)
        + 0.025 * np.maximum(12.0 - hemoglobin, 0)
        + 0.020 * np.maximum(135 - sodium, 0)
        + 0.010 * np.maximum(glucose - 120, 0)
        + 0.05 * np.maximum(followup - 7, 0)
        - 0.018 * (readiness - 70)
        + 0.35 * (admission_type == "emergency")
        + 0.42 * (discharge == "skilled_nursing")
        + 0.24 * (discharge == "home_health")
        + 0.30 * (condition == "renal")
        + rng.normal(0, 0.55, n_samples)
    )

    target = rng.binomial(1, sigmoid(logit))

    df = pd.DataFrame({
        "age": age.round(1),
        "length_of_stay_days": los.round(2),
        "prior_admissions_6m": prior_adm,
        "prior_ed_visits_6m": prior_ed,
        "prior_readmissions_12m": prior_readm,
        "comorbidity_index": comorbidity,
        "medication_count": meds.round(0),
        "abnormal_lab_rate": abnormal_lab_rate.round(4),
        "hemoglobin_g_dl": hemoglobin.round(2),
        "creatinine_mg_dl": creatinine.round(3),
        "sodium_mmol_l": sodium.round(2),
        "glucose_mg_dl": glucose.round(2),
        "followup_days": followup.round(1),
        "discharge_readiness_score": readiness.round(1),
        "admission_type": admission_type,
        "discharge_disposition": discharge,
        "primary_condition_group": condition,
        "readmitted_30d": target,
    })

    for col in [
        "hemoglobin_g_dl", "creatinine_mg_dl", "sodium_mmol_l",
        "glucose_mg_dl", "abnormal_lab_rate", "followup_days"
    ]:
        df.loc[rng.random(n_samples) < 0.015, col] = np.nan

    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def main():
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate_dataset()
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(df):,} synthetic encounters to {DATA_PATH}")
    print(f"30-day readmission prevalence: {df['readmitted_30d'].mean():.3%}")


if __name__ == "__main__":
    main()
