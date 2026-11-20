from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

FEATURES = [f"f{i}" for i in range(12)]


def _add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Raw features plus a few cheap, deterministic interactions."""
    X = df[FEATURES].copy()
    arr = df[FEATURES].to_numpy()
    X["norm"] = np.sqrt((arr ** 2).sum(axis=1))
    X["abs_max"] = np.abs(arr).max(axis=1)
    X["mean"] = arr.mean(axis=1)
    X["std"] = arr.std(axis=1)
    return X


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    data = data.reset_index(drop=True)

    # 1) Split FIRST, stratified, so the test set is untouched, has no
    #    duplicated rows, and keeps the real ~5% fraud rate.
    train, test = train_test_split(
        data, test_size=0.2, random_state=seed, stratify=data["label"]
    )

    X_train, y_train = _add_features(train), train["label"].to_numpy()
    X_test, y_test = _add_features(test), test["label"].to_numpy()

    # 2) Handle imbalance inside the training set only, via sample weights
    #    (no duplicated rows, no leakage).
    n_pos = max(int(y_train.sum()), 1)
    n_neg = len(y_train) - n_pos
    weights = np.where(y_train == 1, n_neg / n_pos, 1.0)
    # Soften the weighting so probabilities stay reasonably calibrated.
    weights = np.sqrt(weights)

    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=400,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=30,
        random_state=seed,
    )
    model.fit(X_train, y_train, sample_weight=weights)
    scores = model.predict_proba(X_test)[:, 1]

    return {
        "train_ids": train["row_id"].astype(int).tolist(),
        "test_ids": test["row_id"].astype(int).tolist(),
        "test_scores": scores.astype(float).tolist(),
        "metric": "average_precision (PR-AUC)",
        "value": float(average_precision_score(y_test, scores)),
        "notes": (
            "Fixed data leakage: the original upsampled fraud rows before "
            "train_test_split, so duplicates appeared in both train and test "
            "and the test set was 50/50 synthetic. Now: stratified split first, "
            "test set left untouched at the natural ~5% fraud rate. Imbalance is "
            "handled only in training via softened (sqrt) class weights instead "
            "of duplication. Replaced RandomForest with HistGradientBoosting "
            "(early stopping, regularized) and added a few aggregate features "
            "(norm, abs max, mean, std). Switched metric from accuracy, which "
            "is ~95% for predicting all-legit, to PR-AUC, which suits rare-"
            "positive fraud detection. Deterministic via seed."
        ),
    }