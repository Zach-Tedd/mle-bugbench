"""Patient-level diagnosis model with leakage-safe evaluation.

The evaluation holds out entire patients, aggregates repeated visits into
one patient-level observation, and reports patient-level ROC-AUC.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [f"f{i}" for i in range(10)]


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    """Train and evaluate a disease prediction model on unseen patients."""

    # ---------------------------------------------------------
    # 1. Validate input data.
    # ---------------------------------------------------------
    required = ["row_id", "patient_id", "label", *FEATURES]
    missing = set(required) - set(data.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if data["row_id"].isna().any() or data["row_id"].duplicated().any():
        raise ValueError("row_id must be non-null and unique.")

    if data["patient_id"].isna().any():
        raise ValueError("patient_id must not contain missing values.")

    if data["label"].isna().any():
        raise ValueError("Labels must not contain missing values.")

    if not set(data["label"].unique()).issubset({0, 1}):
        raise ValueError("This implementation requires binary labels 0 and 1.")

    # Each patient must have exactly one diagnosis.
    label_counts = data.groupby("patient_id")["label"].nunique()

    if (label_counts != 1).any():
        raise ValueError("Each patient must have exactly one diagnosis.")

    patient_labels = data.groupby("patient_id")["label"].first()

    if patient_labels.nunique() != 2:
        raise ValueError("Both classes must be present.")

    # ---------------------------------------------------------
    # 2. Aggregate repeated visits into patient-level features.
    # ---------------------------------------------------------
    # Multiple visits from one patient are correlated observations,
    # not independent samples. Their mean provides one observation
    # per patient and reduces visit-level measurement noise.
    patient_features = (
        data.groupby("patient_id")[FEATURES]
        .mean()
    )

    patient_data = patient_features.join(
        patient_labels.rename("label")
    )

    # ---------------------------------------------------------
    # 3. Split by patient, not by individual visit.
    # ---------------------------------------------------------
    # This prevents visits from the same patient appearing in
    # both training and testing, eliminating patient-level leakage.
    train_patients, test_patients = train_test_split(
        patient_data.index.to_numpy(),
        test_size=0.2,
        stratify=patient_data.loc[:, "label"].to_numpy(),
        random_state=seed,
    )

    train_mask = data["patient_id"].isin(train_patients)
    test_mask = data["patient_id"].isin(test_patients)

    train = data.loc[train_mask]
    test = data.loc[test_mask]

    train_ids = train["row_id"].astype(int).tolist()
    test_ids = test["row_id"].astype(int).tolist()

    # Explicitly verify that the split is patient-independent.
    assert set(train_patients).isdisjoint(set(test_patients))
    assert len(train_ids) + len(test_ids) == len(data)

    # ---------------------------------------------------------
    # 4. Build a patient-level classification pipeline.
    # ---------------------------------------------------------
    # Logistic regression is appropriate for this relatively small
    # dataset and provides probabilities without the unnecessary
    # complexity of a random forest.
    #
    # Imputation and scaling are fitted exclusively on training data.
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                C=1.0,
                max_iter=2000,
                random_state=seed,
            ),
        ),
    ])

    X_train = patient_data.loc[train_patients, FEATURES]
    y_train = patient_data.loc[train_patients, "label"]

    X_test = patient_data.loc[test_patients, FEATURES]
    y_test = patient_data.loc[test_patients, "label"]

    model.fit(X_train, y_train)

    # One prediction per previously unseen patient.
    patient_scores = model.predict_proba(X_test)[:, 1]

    # ---------------------------------------------------------
    # 5. Evaluate at the patient level.
    # ---------------------------------------------------------
    # Each patient contributes exactly once to ROC-AUC, regardless
    # of how many visits they have.
    value = float(
        roc_auc_score(y_test.to_numpy(), patient_scores)
    )

    # ---------------------------------------------------------
    # 6. Return visit-level IDs and corresponding probabilities.
    # ---------------------------------------------------------
    # The required output contains one score per test visit.
    # Every visit belonging to a patient receives that patient's
    # aggregated prediction. This preserves the required ordering
    # and correspondence between test_ids and test_scores.
    score_by_patient = dict(zip(test_patients, patient_scores))

    test_scores = (
        test["patient_id"]
        .map(score_by_patient)
        .astype(float)
        .tolist()
    )

    return {
        "train_ids": train_ids,
        "test_ids": test_ids,
        "test_scores": test_scores,
        "metric": "patient_level_roc_auc",
        "value": value,
        "notes": (
            "The original evaluation randomly split individual visits. "
            "Because patients contribute 3-8 correlated visits, visits "
            "from the same patient could appear in both training and "
            "testing. This caused patient-level data leakage and made "
            "the reported ROC-AUC of 0.93 unreliable. "
            "Additionally, visit-level evaluation gave patients with "
            "more visits disproportionate weight. "
            "The corrected implementation splits by patient, ensuring "
            "that no patient appears in both partitions. It aggregates "
            "each patient's clinical measurements using their visit "
            "mean, trains a standardized logistic regression model "
            "with training-only preprocessing, and evaluates ROC-AUC "
            "once per held-out patient. "
            "test_scores contains the corresponding patient-level "
            "probability repeated for each of that patient's test visits. "
            "The reported metric therefore estimates discrimination "
            "on previously unseen patients rather than previously "
            "unseen visits from familiar patients. "
            "The result is a single holdout estimate, not a confidence "
            "interval or an external validation. Independent external "
            "validation and uncertainty estimates are recommended "
            "before publication."
        ),
    }
