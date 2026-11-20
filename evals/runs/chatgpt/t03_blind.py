from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [f"f{i}" for i in range(10)]


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    data = data.copy()

    # Validate required columns.
    required = ["row_id", "patient_id", "label", *FEATURES]
    missing = set(required) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if data["row_id"].duplicated().any():
        raise ValueError("row_id must uniquely identify each visit.")

    # Each patient must have a consistent disease label.
    patient_labels = data.groupby("patient_id")["label"].nunique()
    if (patient_labels > 1).any():
        raise ValueError("Each patient must have a consistent label.")

    # Split patients rather than visits to prevent information leakage.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=seed,
    )

    train_idx, test_idx = next(
        splitter.split(
            data,
            data["label"],
            groups=data["patient_id"],
        )
    )

    train = data.iloc[train_idx].copy()
    test = data.iloc[test_idx].copy()

    # Aggregate visits into patient-level statistics.
    # The patient-specific offset is shared across visits, so the
    # patient mean is a useful representation of the underlying signal.
    patient_features = (
        data.groupby("patient_id")[FEATURES]
        .agg(["mean", "std"])
    )

    patient_features.columns = [
        f"{feature}_{stat}"
        for feature, stat in patient_features.columns
    ]

    patient_features = patient_features.reset_index()

    # One training observation per patient.
    train_patients = train["patient_id"].drop_duplicates()
    test_patients = test["patient_id"].drop_duplicates()

    X_train = (
        patient_features[
            patient_features["patient_id"].isin(train_patients)
        ]
        .set_index("patient_id")
        .loc[train_patients]
    )

    y_train = (
        train.groupby("patient_id")["label"]
        .first()
        .loc[train_patients]
    )

    X_test = (
        patient_features[
            patient_features["patient_id"].isin(test_patients)
        ]
        .set_index("patient_id")
        .loc[test_patients]
    )

    y_test = (
        test.groupby("patient_id")["label"]
        .first()
        .loc[test_patients]
    )

    # Regularized logistic regression.
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                C=0.1,
                class_weight=None,
                max_iter=2000,
                solver="lbfgs",
                random_state=seed,
            ),
        ),
    ])

    model.fit(X_train, y_train)

    # Predict once per patient.
    patient_scores = model.predict_proba(X_test)[:, 1]

    scores_by_patient = dict(
        zip(test_patients, patient_scores)
    )

    # Return one prediction for every test visit.
    scores = test["patient_id"].map(scores_by_patient).to_numpy()

    # Evaluate at the patient level so each patient contributes equally.
    value = float(roc_auc_score(y_test, patient_scores))

    return {
        "train_ids": train["row_id"].astype(int).tolist(),
        "test_ids": test["row_id"].astype(int).tolist(),
        "test_scores": scores.astype(float).tolist(),
        "metric": "patient_level_roc_auc",
        "value": value,
        "notes": (
            "Replaced visit-level random splitting with "
            "GroupShuffleSplit by patient_id to prevent patient "
            "leakage. Aggregated visits into patient-level means "
            "and standard deviations, capturing the shared "
            "patient-specific signal while reducing visit noise. "
            "Used median imputation, standardization, and "
            "regularized logistic regression. Predictions are "
            "generated once per patient and mapped back to every "
            "test visit. ROC AUC is evaluated at the patient level."
        ),
    }