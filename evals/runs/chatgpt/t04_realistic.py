
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


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    """
    Train and evaluate an ensemble of regularized classifiers.

    Feature selection and preprocessing are performed independently
    inside each training fold to prevent validation leakage.

    Returns:
        folds: Out-of-fold predictions and row IDs.
        metric: Evaluation metric.
        value: Mean cross-validation ROC-AUC.
        notes: Description of the modeling approach.
    """

    # ---------------------------------------------------------
    # 1. Prepare data
    # ---------------------------------------------------------

    feature_cols = [
        c for c in data.columns
        if c.startswith("g") and c[1:].isdigit()
    ]

    if not feature_cols:
        raise ValueError("No feature columns matching g<number> were found.")

    x = data[feature_cols].to_numpy(dtype=np.float64)
    y = data["label"].to_numpy(dtype=int)
    row_ids = data["row_id"].tolist()

    if len(np.unique(y)) != 2:
        raise ValueError("Binary classification requires exactly two classes.")

    if not np.isfinite(x).all():
        # Imputation is also performed within each fold.
        pass

    n_samples, n_features = x.shape

    if n_samples < 5:
        raise ValueError("At least five samples are required.")

    # ---------------------------------------------------------
    # 2. Ensemble configurations
    # ---------------------------------------------------------

    # Different feature counts help reduce dependence on one
    # potentially unstable feature-selection cutoff.
    feature_counts = [10, 25, 50, 100]

    # Stronger regularization is generally useful when p >> n.
    regularization = [0.03, 0.3]

    configurations = [
        (k, c)
        for k in feature_counts
        for c in regularization
    ]

    # ---------------------------------------------------------
    # 3. Cross-validation
    # ---------------------------------------------------------

    n_splits = min(5, int(np.bincount(y).min()))

    if n_splits < 2:
        raise ValueError("Each class must contain at least two samples.")

    splitter = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    folds = []
    aucs = []

    validation_counts = np.zeros(n_samples, dtype=int)

    for fold_idx, (train_idx, val_idx) in enumerate(
        splitter.split(x, y)
    ):

        x_train = x[train_idx]
        x_val = x[val_idx]

        y_train = y[train_idx]
        y_val = y[val_idx]

        validation_counts[val_idx] += 1

        # -----------------------------------------------------
        # 4. Train ensemble
        # -----------------------------------------------------

        predictions = []

        for k, c in configurations:

            selected_k = min(k, n_features)

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
                        k=selected_k,
                    ),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        C=c,
                        penalty="l2",
                        solver="liblinear",
                        max_iter=2000,
                        random_state=seed + fold_idx,
                    ),
                ),
            ])

            model.fit(x_train, y_train)

            probabilities = model.predict_proba(x_val)[:, 1]

            predictions.append(probabilities)

        # Average ensemble probabilities.
        scores = np.mean(
            np.stack(predictions, axis=0),
            axis=0,
        )

        # -----------------------------------------------------
        # 5. Evaluate fold
        # -----------------------------------------------------

        fold_auc = roc_auc_score(y_val, scores)
        aucs.append(fold_auc)

        folds.append({
            "train_ids": [
                row_ids[i] for i in train_idx
            ],
            "val_ids": [
                row_ids[i] for i in val_idx
            ],
            "val_scores": scores.tolist(),
        })

    # ---------------------------------------------------------
    # 6. Validate out-of-fold coverage
    # ---------------------------------------------------------

    if not np.all(validation_counts == 1):
        raise RuntimeError(
            "Cross-validation error: every sample must be "
            "validated exactly once."
        )

    # ---------------------------------------------------------
    # 7. Return results
    # ---------------------------------------------------------

    return {
        "folds": folds,
        "metric": "roc_auc",
        "value": float(np.mean(aucs)),
        "notes": (
            "Fixed feature-selection leakage by moving all "
            "preprocessing and supervised feature selection "
            "inside each training fold. Replaced the single "
            "logistic regression model with an ensemble of "
            "eight L2-regularized logistic regression models "
            "using feature-selection sizes of 10, 25, 50, "
            "and 100 and regularization strengths of 0.03 "
            "and 0.3. StandardScaler and median imputation "
            "are fitted exclusively on training data. "
            "Predictions are averaged across ensemble members. "
            "Stratified K-fold validation, deterministic seeds, "
            "and exactly-once validation coverage are enforced."
        ),
    }
