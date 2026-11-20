
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [f"g{i}" for i in range(2000)]

# Multiple feature-selection sizes help reduce sensitivity
# to the number of informative features.
FEATURE_COUNTS = (10, 20, 40, 80)

# Moderate regularization is appropriate for p >> n.
REGULARIZATION = 0.1


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    """
    Train and evaluate an ensemble using stratified 5-fold CV.

    Returns:
        folds: Validation predictions and corresponding row IDs.
        metric: Evaluation metric name.
        value: Mean validation ROC AUC.
        notes: Description of modeling improvements.
    """

    x = data[FEATURES].to_numpy(dtype=np.float64)
    y = data["label"].to_numpy(dtype=int)
    row_ids = data["row_id"].to_numpy()

    if len(np.unique(y)) != 2:
        raise ValueError("Binary classification requires exactly two classes.")

    if len(np.unique(row_ids)) != len(row_ids):
        raise ValueError("row_id values must be unique.")

    splitter = StratifiedKFold(
        n_splits=5,
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

        ensemble_scores = []

        for k in FEATURE_COUNTS:
            k = min(k, x_train.shape[1])

            # Feature selection is fitted ONLY on training data.
            selector = SelectKBest(
                score_func=f_classif,
                k=k,
            )

            x_train_selected = selector.fit_transform(
                x_train,
                y_train,
            )

            x_val_selected = selector.transform(x_val)

            # Scaling is also fitted exclusively on training data.
            model = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=REGULARIZATION,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=seed,
                ),
            )

            model.fit(x_train_selected, y_train)

            scores = model.predict_proba(
                x_val_selected
            )[:, 1]

            ensemble_scores.append(scores)

        # Average probabilities from models with different
        # feature-selection sizes.
        val_scores = np.mean(
            np.vstack(ensemble_scores),
            axis=0,
        )

        fold_auc = roc_auc_score(
            y_val,
            val_scores,
        )

        aucs.append(fold_auc)

        folds.append(
            {
                "train_ids": row_ids[train_idx].tolist(),
                "val_ids": row_ids[val_idx].tolist(),
                "val_scores": val_scores.tolist(),
            }
        )

    return {
        "folds": folds,
        "metric": "roc_auc",
        "value": float(np.mean(aucs)),
        "notes": (
            "Fixed feature-selection leakage by fitting SelectKBest "
            "independently within each training fold. Introduced an "
            "ensemble of L2-regularized logistic regression models "
            "using 10, 20, 40, and 80 selected features. StandardScaler "
            "is fitted separately within each fold to prevent leakage. "
            "Validation probabilities are averaged across ensemble "
            "members. Stratified five-fold cross-validation and "
            "deterministic seeding are preserved."
        ),
    }
