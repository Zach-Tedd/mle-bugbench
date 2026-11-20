
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    """
    Train and evaluate a failure prediction model.

    Returns:
        train_ids: row_ids used for training and model selection.
        test_ids: held-out row_ids.
        test_scores: predicted failure probabilities.
        metric: evaluation metric.
        value: held-out test ROC AUC.
        notes: description of the modeling approach.
    """

    # ---------------------------------------------------------
    # 1. Validate input
    # ---------------------------------------------------------
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")

    required = {"row_id", "label"}
    missing = required.difference(data.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if data["row_id"].isna().any():
        raise ValueError("row_id cannot contain missing values")

    if data["row_id"].duplicated().any():
        raise ValueError("row_id must be unique")

    if data["label"].isna().any():
        raise ValueError("label cannot contain missing values")

    labels = data["label"].to_numpy()

    if not np.isin(labels, [0, 1]).all():
        raise ValueError("label must contain only binary values 0 and 1")

    if len(np.unique(labels)) != 2:
        raise ValueError("Both classes must be present")

    # Use all available sensor columns.
    feature_cols = [
        col for col in data.columns
        if col not in ("row_id", "label")
    ]

    if not feature_cols:
        raise ValueError("No sensor features found")

    X = data[feature_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    X = X.replace([np.inf, -np.inf], np.nan)

    if X.isna().all(axis=0).any():
        raise ValueError("At least one feature contains no valid values")

    y = data["label"].astype(int)

    # ---------------------------------------------------------
    # 2. Stratified machine-level holdout
    # ---------------------------------------------------------
    indices = np.arange(len(data))

    train_idx, test_idx = train_test_split(
        indices,
        test_size=0.30,
        stratify=y,
        random_state=seed,
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]

    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    # ---------------------------------------------------------
    # 3. Build leakage-safe modeling pipeline
    # ---------------------------------------------------------
    #
    # Imputation, scaling, feature selection and classification
    # are all fitted independently inside each CV fold.
    #
    # SelectKBest uses the training-fold data only.
    #
    # Logistic regression provides regularization to reduce
    # overfitting in this relatively small dataset.
    #

    max_features = len(feature_cols)

    candidate_k = sorted({
        min(k, max_features)
        for k in [3, 5, 8, 12, 20, 30]
    })

    candidate_c = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=seed,
    )

    best_auc = -np.inf
    best_params = None
    best_model = None

    # ---------------------------------------------------------
    # 4. Hyperparameter selection using training data only
    # ---------------------------------------------------------
    for k in candidate_k:
        for c in candidate_c:

            model = Pipeline([
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "selector",
                    SelectKBest(
                        score_func=f_classif,
                        k=k,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        C=c,
                        max_iter=2000,
                        solver="lbfgs",
                        random_state=seed,
                    ),
                ),
            ])

            fold_scores = cross_val_score(
                model,
                X_train,
                y_train,
                cv=cv,
                scoring="roc_auc",
                n_jobs=1,
                error_score="raise",
            )

            mean_auc = float(np.mean(fold_scores))

            # Strict comparison ensures deterministic tie-breaking.
            if mean_auc > best_auc:
                best_auc = mean_auc
                best_params = {
                    "k": k,
                    "C": c,
                }
                best_model = model

    if best_model is None:
        raise RuntimeError("Model selection failed")

    # ---------------------------------------------------------
    # 5. Refit selected model on all training observations
    # ---------------------------------------------------------
    best_model.fit(X_train, y_train)

    # ---------------------------------------------------------
    # 6. Evaluate held-out test observations exactly once
    # ---------------------------------------------------------
    test_scores = best_model.predict_proba(X_test)[:, 1]

    test_scores = np.clip(
        test_scores,
        0.0,
        1.0,
    )

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

        "notes": (
            "Replaced test-set feature-subset selection with "
            "leakage-safe stratified 5-fold cross-validation. "
            "Uses median imputation, standardization, "
            "training-fold univariate feature selection, and "
            "L2-regularized logistic regression. Tunes the "
            "number of selected features and regularization "
            "strength using training data only. Refits the "
            "selected pipeline on the complete training set "
            "and evaluates the held-out 30% test set once. "
            f"Selected {best_params['k']} features with "
            f"C={best_params['C']}."
        ),
    }
