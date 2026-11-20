"""Failure prediction with leakage-free model selection and honest test evaluation.

The 200 candidate sensor subsets are selected using cross-validation on the
training set only. The held-out test set is used exactly once, after selection,
to estimate performance on new machines.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Load the original, unchanged candidate subsets.
_spec = importlib.util.spec_from_file_location(
    "t05_data",
    Path(__file__).parents[1] / "data.py",
)
assert _spec is not None and _spec.loader is not None

_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_data)

CANDIDATES: list[list[str]] = _data.candidate_subsets()


def make_model(seed: int):
    """Create a model with preprocessing fitted only on its training data."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            random_state=seed,
        ),
    )


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    """Select among 200 sensor subsets and estimate performance on new machines."""

    # Separate the data before any model selection.
    # The test set must never influence candidate selection.
    train, test = train_test_split(
        data,
        test_size=0.30,
        stratify=data["label"],
        random_state=seed,
    )

    train = train.copy()
    test = test.copy()

    X_train = train.drop(columns=["row_id", "label"])
    y_train = train["label"].to_numpy()

    X_test = test.drop(columns=["row_id", "label"])
    y_test = test["label"].to_numpy()

    # Inner cross-validation: all 200 candidates are evaluated using
    # training data only. Every preprocessing operation is fitted
    # independently inside each fold.
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )

    best_auc = -np.inf
    best_cols = None

    for cols in CANDIDATES:
        fold_aucs = []

        for fold_train_idx, fold_val_idx in cv.split(X_train, y_train):
            X_fold_train = X_train.iloc[fold_train_idx][cols]
            y_fold_train = y_train[fold_train_idx]

            X_fold_val = X_train.iloc[fold_val_idx][cols]
            y_fold_val = y_train[fold_val_idx]

            model = make_model(seed)

            model.fit(X_fold_train, y_fold_train)

            val_scores = model.predict_proba(X_fold_val)[:, 1]

            fold_aucs.append(
                roc_auc_score(y_fold_val, val_scores)
            )

        # Select using mean validation ROC-AUC, never test ROC-AUC.
        mean_auc = float(np.mean(fold_aucs))

        # Strict comparison provides deterministic first-candidate
        # tie-breaking.
        if mean_auc > best_auc:
            best_auc = mean_auc
            best_cols = cols

    if best_cols is None:
        raise RuntimeError("No candidate sensor subset was evaluated.")

    # Refit the selected model using all available training machines.
    # The test set remains completely untouched until this point.
    final_model = make_model(seed)

    final_model.fit(
        X_train[best_cols],
        y_train,
    )

    # Evaluate exactly once on the independent held-out test machines.
    test_scores = final_model.predict_proba(
        X_test[best_cols]
    )[:, 1]

    test_auc = roc_auc_score(y_test, test_scores)

    return {
        "train_ids": train["row_id"].astype(int).tolist(),
        "test_ids": test["row_id"].astype(int).tolist(),
        "test_scores": test_scores.astype(float).tolist(),
        "metric": "roc_auc",
        "value": float(test_auc),
        "notes": (
            "The original implementation evaluated 200 candidate models "
            "on the test set and reported the maximum test ROC-AUC. "
            "This causes substantial selection bias: the test set was "
            "effectively used for model selection, so the reported 0.74 "
            "is an optimistically biased performance estimate. "
            "The corrected implementation preserves all 200 original "
            "candidate sensor subsets but selects among them using "
            "5-fold stratified cross-validation exclusively within the "
            "training set. StandardScaler is fitted separately inside "
            "each fold to prevent preprocessing leakage. The selected "
            "LogisticRegression model is then refitted on the entire "
            "training set and evaluated once on an independent, "
            "stratified 30% test set. The returned ROC-AUC is therefore "
            "an honest held-out estimate of discrimination on new "
            "machines, subject to sampling uncertainty from the "
            "relatively small dataset. No test labels or test scores "
            "influence candidate selection."
        ),
    }