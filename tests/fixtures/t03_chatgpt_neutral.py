"""Leakage-resistant clinical diagnosis model.

Patients, not visits, are the independent evaluation units.
All visits belonging to a patient are assigned to the same split.

The model is trained on patient-level aggregated measurements.
Evaluation uses one prediction per held-out patient.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [f"f{i}" for i in range(10)]


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    """Train and evaluate a patient-level disease prediction model."""

    required = ["row_id", "patient_id", "label", *FEATURES]
    missing = set(required) - set(data.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if data["row_id"].duplicated().any():
        raise ValueError("row_id must uniquely identify visits.")

    if data[required].isna().any().any():
        raise ValueError("Missing values must be handled before evaluation.")

    if not set(data["label"].unique()).issubset({0, 1}):
        raise ValueError("label must contain binary values 0 and 1.")

    # ---------------------------------------------------------
    # 1. Validate patient-level labels.
    # ---------------------------------------------------------

    patient_labels = data.groupby("patient_id")["label"].nunique()

    if (patient_labels != 1).any():
        raise ValueError(
            "Each patient must have exactly one consistent diagnosis."
        )

    # One observation per patient for splitting and evaluation.
    patient_table = (
        data.groupby("patient_id", sort=True)["label"]
        .first()
        .reset_index()
    )

    if patient_table["label"].nunique() != 2:
        raise ValueError("Both diagnosis classes are required.")

    # ---------------------------------------------------------
    # 2. Stratified patient-level split.
    # ---------------------------------------------------------

    train_patients, test_patients = train_test_split(
        patient_table["patient_id"].to_numpy(),
        test_size=0.2,
        stratify=patient_table["label"].to_numpy(),
        random_state=seed,
    )

    train_patients = set(train_patients)
    test_patients = set(test_patients)

    if train_patients & test_patients:
        raise RuntimeError("Patient leakage detected.")

    train = data[data["patient_id"].isin(train_patients)].copy()
    test = data[data["patient_id"].isin(test_patients)].copy()

    # ---------------------------------------------------------
    # 3. Aggregate repeated visits into patient-level features.
    #
    # The mean reduces visit-level measurement noise.
    # Each patient contributes exactly one training example,
    # preventing patients with more visits from receiving
    # disproportionately high training weight.
    # ---------------------------------------------------------

    train_agg = (
        train.groupby("patient_id", sort=True)[FEATURES]
        .mean()
    )

    test_agg = (
        test.groupby("patient_id", sort=True)[FEATURES]
        .mean()
    )

    y_train = (
        train.groupby("patient_id", sort=True)["label"]
        .first()
        .loc[train_agg.index]
    )

    y_test = (
        test.groupby("patient_id", sort=True)["label"]
        .first()
        .loc[test_agg.index]
    )

    # ---------------------------------------------------------
    # 4. Standardized regularized logistic regression.
    #
    # Scaling is fitted exclusively on training patients.
    # L2 regularization helps control overfitting with a
    # relatively small number of independent patients.
    # ---------------------------------------------------------

    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                C=1.0,
                penalty="l2",
                solver="lbfgs",
                max_iter=2000,
                random_state=seed,
            ),
        ),
    ])

    model.fit(train_agg, y_train)

    # ---------------------------------------------------------
    # 5. Predict one probability per held-out patient.
    # ---------------------------------------------------------

    patient_scores = model.predict_proba(test_agg)[:, 1]

    patient_predictions = pd.Series(
        patient_scores,
        index=test_agg.index,
    )

    # Map patient-level predictions back to every held-out visit.
    # Thus every test row receives the same score as its patient.
    visit_scores = test["patient_id"].map(patient_predictions)

    # ---------------------------------------------------------
    # 6. Evaluate at the independent patient level.
    # ---------------------------------------------------------

    value = roc_auc_score(y_test, patient_scores)

    # Preserve original visit ordering and ensure each row_id
    # is returned exactly once.
    test = test.reset_index(drop=True)

    scores_by_patient = test["patient_id"].map(patient_predictions)

    return {
        "train_ids": train["row_id"].astype(int).tolist(),
        "test_ids": test["row_id"].astype(int).tolist(),
        "test_scores": scores_by_patient.astype(float).tolist(),
        "metric": "patient_level_roc_auc",
        "value": float(value),
        "notes": (
            "Replaced random visit-level splitting with stratified "
            "patient-level splitting to prevent leakage between "
            "correlated visits. Aggregated each patient's visits "
            "using feature means, ensuring one training example "
            "per patient. Replaced the random forest with a "
            "StandardScaler and L2-regularized logistic regression "
            "pipeline. Scaling is fitted only on training patients. "
            "ROC-AUC is calculated using one prediction per held-out "
            "patient, so each patient contributes equally. Visit-level "
            "test scores are mapped back to their corresponding "
            "patients. The previously reported visit-level ROC-AUC "
            "is not directly comparable because it may be inflated "
            "by patient-level data leakage. This evaluation is a "
            "single held-out patient split, not external validation."
        ),
    }
