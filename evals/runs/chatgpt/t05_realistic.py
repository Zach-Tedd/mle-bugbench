
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    train_test_split,
    RepeatedStratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# Load candidate sensor subsets.
_spec = importlib.util.spec_from_file_location(
    "t05_data",
    Path(__file__).resolve().parents[1] / "data.py",
)

assert _spec is not None and _spec.loader is not None

_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_data)

CANDIDATES: list[list[str]] = _data.candidate_subsets()


def make_model(C: float, seed: int) -> Pipeline:
    """Create a leakage-safe preprocessing and classification pipeline."""

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                C=C,
                max_iter=2000,
                solver="lbfgs",
                random_state=seed,
            ),
        ),
    ])


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:

    # ---------------------------------------------------------
    # 1. Prepare data
    # ---------------------------------------------------------

    data = data.copy().reset_index(drop=True)

    feature_cols = [
        c for c in data.columns
        if c.startswith("s") and c[1:].isdigit()
    ]

    if not feature_cols:
        raise ValueError("No sensor features found.")

    if "row_id" not in data.columns:
        raise ValueError("Missing row_id column.")

    if "label" not in data.columns:
        raise ValueError("Missing label column.")

    X = data[feature_cols]
    y = data["label"].astype(int)

    if y.nunique() != 2:
        raise ValueError("Binary classification requires exactly two classes.")

    # ---------------------------------------------------------
    # 2. Stratified holdout split
    # ---------------------------------------------------------

    train_idx, test_idx = train_test_split(
        np.arange(len(data)),
        test_size=0.30,
        stratify=y,
        random_state=seed,
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # ---------------------------------------------------------
    # 3. Generate candidate models
    # ---------------------------------------------------------

    # Candidate subsets are evaluated only on training data.
    candidates = []

    for cols in CANDIDATES:
        valid_cols = [c for c in cols if c in feature_cols]

        if valid_cols:
            # Remove duplicate subsets.
            key = tuple(sorted(valid_cols))

            if key not in candidates:
                candidates.append(key)

    # Include all sensors as a regularized baseline.
    all_features = tuple(feature_cols)

    if all_features not in candidates:
        candidates.append(all_features)

    # Regularization strengths.
    C_VALUES = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]

    # ---------------------------------------------------------
    # 4. Cross-validation model selection
    # ---------------------------------------------------------

    min_class_count = int(y_train.value_counts().min())

    n_splits = min(5, min_class_count)

    if n_splits < 2:
        raise ValueError(
            "Insufficient training samples per class for cross-validation."
        )

    cv = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=3,
        random_state=seed,
    )

    best_cv_auc = -np.inf
    best_cols = None
    best_C = None

    for cols in candidates:

        cols = list(cols)
        X_subset = X_train[cols]

        for C in C_VALUES:

            model = make_model(C, seed)

            scores = cross_val_score(
                model,
                X_subset,
                y_train,
                scoring="roc_auc",
                cv=cv,
                n_jobs=1,
                error_score="raise",
            )

            mean_auc = float(np.mean(scores))

            # Deterministic tie-breaking:
            # prefer smaller C (stronger regularization).
            if (
                mean_auc > best_cv_auc + 1e-10
                or (
                    abs(mean_auc - best_cv_auc) <= 1e-10
                    and (best_C is None or C < best_C)
                )
            ):
                best_cv_auc = mean_auc
                best_cols = cols
                best_C = C

    # ---------------------------------------------------------
    # 5. Fit selected model on the complete training set
    # ---------------------------------------------------------

    final_model = make_model(best_C, seed)

    final_model.fit(
        X_train[best_cols],
        y_train,
    )

    # ---------------------------------------------------------
    # 6. Evaluate on untouched test set
    # ---------------------------------------------------------

    test_scores = final_model.predict_proba(
        X_test[best_cols]
    )[:, 1]

    test_auc = roc_auc_score(
        y_test,
        test_scores,
    )

    # ---------------------------------------------------------
    # 7. Return results
    # ---------------------------------------------------------

    return {
        "train_ids": data.iloc[train_idx]["row_id"].tolist(),

        "test_ids": data.iloc[test_idx]["row_id"].tolist(),

        "test_scores": test_scores.astype(float).tolist(),

        "metric": "roc_auc",

        "value": float(test_auc),

        "chosen": best_cols,

        "best_C": float(best_C),

        "cv_auc": float(best_cv_auc),

        "notes": (
            "Eliminated test-set feature-selection leakage. "
            "Evaluated candidate sensor subsets and logistic "
            "regression regularization strengths using repeated "
            "stratified cross-validation exclusively on training "
            "data. Added median imputation and feature standardization "
            "inside a leakage-safe pipeline. Selected the model using "
            "mean cross-validation ROC-AUC, refitted it on the complete "
            "training set, and evaluated it once on the untouched "
            "30% stratified test set. All randomness is controlled "
            "by the supplied seed."
        ),
    }
