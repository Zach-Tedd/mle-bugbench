"""Geographically robust crop-disease classification across five Kenyan counties.

Improvements:
- Leave-one-county-out cross-validation to measure geographic generalization.
- Exclude f3, which contains a documented Nyeri-specific spurious shade cue.
- Focus on f0-f2, the features carrying the shared disease signal.
- Use standardized, regularized logistic regression for robust generalization.
- Fit preprocessing independently within each training fold.
- Report mean ROC-AUC across all five held-out counties.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURES = [f"f{i}" for i in range(8)]

# f0-f2 carry the disease signal across all counties.
# f3 contains a documented county-specific shade artifact.
MODEL_FEATURES = ["f0", "f1", "f2"]


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    """Train and evaluate a geographically robust disease classifier.

    Each fold holds out one entire county. Predictions are therefore
    evaluated on geographically unseen data rather than randomly
    sampled images from counties already represented in training.

    Args:
        data: DataFrame containing row_id, region, label and image features.
        seed: Random seed for deterministic model fitting.

    Returns:
        Dictionary containing validation folds, mean ROC-AUC and notes.
    """

    required_columns = [
        "row_id",
        "region",
        "label",
        *MODEL_FEATURES,
    ]

    missing = set(required_columns) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if data["row_id"].duplicated().any():
        raise ValueError("row_id must contain unique identifiers.")

    if data["region"].isna().any():
        raise ValueError("region cannot contain missing values.")

    if data["label"].isna().any():
        raise ValueError("label cannot contain missing values.")

    if data[MODEL_FEATURES].isna().any().any():
        raise ValueError("Model features cannot contain missing values.")

    if data["label"].nunique() != 2:
        raise ValueError("Binary classification requires both label classes.")

    X = data[MODEL_FEATURES].astype(float)
    y = data["label"].astype(int)
    groups = data["region"]

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("Labels must be binary (0 and 1).")

    # Each county serves as an independent geographic validation fold.
    splitter = LeaveOneGroupOut()

    folds = []
    fold_aucs = []

    for train_idx, val_idx in splitter.split(X, y, groups):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        # Scaling is fitted only on training counties to prevent leakage.
        model = Pipeline(
            steps=[
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
            ]
        )

        model.fit(X_train, y_train)

        # Probability of disease (positive class).
        scores = model.predict_proba(X_val)[:, 1]

        auc = float(roc_auc_score(y_val, scores))
        fold_aucs.append(auc)

        folds.append(
            {
                "train_ids": data.iloc[train_idx]["row_id"].tolist(),
                "val_ids": data.iloc[val_idx]["row_id"].tolist(),
                "val_scores": scores.tolist(),
            }
        )

    mean_auc = float(np.mean(fold_aucs))

    notes = (
        "Replaced the single-county random split with five-fold "
        "leave-one-county-out cross-validation. Each fold trains on "
        "four counties and validates on the geographically unseen "
        "fifth county. Excluded f3 because it contains a documented "
        "Nyeri-specific spurious shade cue that could inflate random "
        "Nyeri validation performance and undermine national rollout. "
        "Retained f0-f2, which carry the shared disease signal across "
        "counties, and excluded f4-f7 as noise features. "
        "Used standardized L2-regularized logistic regression instead "
        "of a random forest to provide a simpler, more regularized "
        "model with better geographic generalization potential. "
        "StandardScaler is fitted separately within each training fold "
        "to prevent preprocessing leakage. The reported ROC-AUC is "
        "the unweighted mean of the five county-level validation AUCs."
    )

    return {
        "folds": folds,
        "metric": "roc_auc",
        "value": mean_auc,
        "notes": notes,
    }