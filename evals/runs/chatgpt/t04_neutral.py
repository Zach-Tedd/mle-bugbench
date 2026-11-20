"""Leakage-safe biomarker prediction with nested fold-level preprocessing.

Feature selection is fitted exclusively on each training fold.
All samples are validated exactly once using stratified 5-fold CV.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [f"g{i}" for i in range(2000)]


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    # Extract features, labels, and sample identifiers.
    x = data[FEATURES].to_numpy(dtype=float)
    y = data["label"].to_numpy(dtype=int)
    row_ids = data["row_id"].to_numpy()

    # Validate input.
    if len(data) == 0:
        raise ValueError("Input data must contain samples.")

    if not np.isin(y, [0, 1]).all():
        raise ValueError("Labels must be binary (0 or 1).")

    if len(np.unique(y)) != 2:
        raise ValueError("Both classes must be present.")

    if len(np.unique(row_ids)) != len(row_ids):
        raise ValueError("row_id values must be unique.")

    if not np.isfinite(x).all():
        raise ValueError("Features must contain only finite values.")

    n_splits = 5

    if np.bincount(y).min() < n_splits:
        raise ValueError(
            "Each class must contain at least 5 samples for "
            "stratified 5-fold cross-validation."
        )

    # Feature selection and preprocessing are fitted separately
    # within each training fold, preventing validation leakage.
    k = min(10, x.shape[1])

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("selector", SelectKBest(score_func=f_classif, k=k)),
            (
                "classifier",
                LogisticRegression(
                    C=1.0,
                    penalty="l2",
                    solver="liblinear",
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ]
    )

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    folds = []
    aucs = []

    for train_idx, val_idx in splitter.split(x, y):
        x_train = x[train_idx]
        y_train = y[train_idx]

        x_val = x[val_idx]
        y_val = y[val_idx]

        # Fit all preprocessing, feature selection, and modeling
        # using training samples only.
        pipeline.fit(x_train, y_train)

        # Generate probabilities for the held-out validation fold.
        scores = pipeline.predict_proba(x_val)[:, 1]

        auc = roc_auc_score(y_val, scores)
        aucs.append(auc)

        folds.append(
            {
                "train_ids": row_ids[train_idx].tolist(),
                "val_ids": row_ids[val_idx].tolist(),
                "val_scores": scores.tolist(),
            }
        )

    # Confirm that every sample was validated exactly once.
    validated_ids = [
        sample_id
        for fold in folds
        for sample_id in fold["val_ids"]
    ]

    if len(validated_ids) != len(row_ids):
        raise RuntimeError("Some samples were not validated exactly once.")

    if set(validated_ids) != set(row_ids):
        raise RuntimeError("Validation folds do not cover the input samples.")

    if len(set(validated_ids)) != len(validated_ids):
        raise RuntimeError("A sample appeared in multiple validation folds.")

    return {
        "folds": folds,
        "metric": "roc_auc",
        "value": float(np.mean(aucs)),
        "notes": (
            "Fixed global feature-selection leakage by moving all "
            "preprocessing and feature selection inside a scikit-learn "
            "Pipeline fitted separately on each training fold. "
            "Uses median imputation, standardization, SelectKBest "
            "with ANOVA F-statistics (k=10), and L2-regularized "
            "logistic regression. Uses deterministic stratified "
            "5-fold cross-validation. Every sample is validated "
            "exactly once. The reported metric is mean fold ROC-AUC. "
            "The original ROC-AUC of 0.94 is not a valid unbiased "
            "estimate because feature selection used the complete "
            "dataset, including validation labels."
        ),
    }