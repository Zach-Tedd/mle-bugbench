
"""Leakage-resistant fraud classification with honest evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit


FEATURES = [f"f{i}" for i in range(12)]


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    # Work with the original dataset. Never oversample before splitting.
    data = data.copy().reset_index(drop=True)

    if not set(FEATURES + ["label", "row_id"]).issubset(data.columns):
        raise ValueError("Dataset must contain row_id, label and all 12 features.")

    if data["row_id"].duplicated().any():
        raise ValueError("row_id must uniquely identify each dataset row.")

    if data["label"].nunique() != 2:
        raise ValueError("Both fraud and non-fraud classes must be present.")

    if data[FEATURES].isna().any().any():
        raise ValueError("Feature columns contain missing values.")

    X = data[FEATURES]
    y = data["label"].astype(int)

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("Labels must be binary (0 and 1).")

    # Group identical feature vectors together.
    # This prevents exact duplicate transactions from crossing
    # the training/test boundary, even if their row_ids differ.
    groups = pd.factorize(
        pd.MultiIndex.from_frame(X)
    )[0]

    # Split BEFORE fitting or applying any sampling strategy.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=seed,
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    X_test = X.iloc[test_idx]
    y_test = y.iloc[test_idx]

    # Class weighting handles imbalance without artificially
    # duplicating minority-class observations.
    model = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # Evaluate probabilities on untouched, naturally distributed data.
    scores = model.predict_proba(X_test)[:, 1]

    # Average Precision is more informative than accuracy when
    # fraud is rare, since it focuses on precision and recall.
    value = average_precision_score(y_test, scores)

    return {
        "train_ids": data.iloc[train_idx]["row_id"].astype(int).tolist(),
        "test_ids": data.iloc[test_idx]["row_id"].astype(int).tolist(),
        "test_scores": scores.astype(float).tolist(),
        "metric": "average_precision",
        "value": float(value),
        "notes": (
            "Removed data leakage caused by oversampling before splitting. "
            "Split the original dataset using GroupShuffleSplit, keeping "
            "identical feature vectors in the same partition to prevent "
            "duplicate transaction leakage. Excluded row_id from model "
            "features. Replaced random oversampling with "
            "class_weight='balanced_subsample'. Evaluated on the untouched "
            "test set with its natural class distribution. Replaced "
            "accuracy with Average Precision (PR-AUC), which is more "
            "appropriate for imbalanced fraud detection. The reported "
            "score is a held-out estimate, not a guarantee of production "
            "performance."
        ),
    }
