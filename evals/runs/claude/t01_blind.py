from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

FEATURES = [f"f{i}" for i in range(12)]


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    # 1. Remove exact duplicate observations (same features and label) so the
    #    same transaction cannot appear in both train and test.
    #    Sorting by row_id first makes the kept row deterministic.
    df = data.sort_values("row_id")
    n_before = len(df)
    df = df.drop_duplicates(subset=FEATURES + ["label"], keep="first")
    n_dupes = n_before - len(df)

    # 2. Split FIRST, stratified, keeping the natural class ratio in the test set.
    train, test = train_test_split(
        df,
        test_size=0.25,
        random_state=seed,
        stratify=df["label"],
    )

    # 3. Fit on the training split only. Imbalance is handled with class
    #    weights rather than by duplicating rows (no leakage, no repeats).
    model = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(train[FEATURES], train["label"])
    scores = model.predict_proba(test[FEATURES])[:, 1]

    value = float(average_precision_score(test["label"], scores))

    notes = (
        f"Changes: (1) Dropped {n_dupes} exact duplicate rows before splitting, "
        "since duplicates (mostly fraud) leaked identical rows into both train "
        "and test. (2) Split before any rebalancing and stratified by label; the "
        "test set keeps the real ~3% fraud rate and is never resampled. The "
        "original oversampled positives before the split, which put copies of "
        "the same fraud rows in train and test and inflated results. (3) "
        "Replaced upsampling with class_weight='balanced' on a "
        "HistGradientBoosting model (regularized, early stopping on an internal "
        "validation split of the training data), so train_ids has no repeats. "
        "(4) Replaced accuracy with average precision (PR-AUC): at this "
        "prevalence a model that never flags fraud scores ~97% accuracy. "
        "Scores are raw probabilities under class weighting, so they are "
        "ranking-calibrated, not true fraud probabilities; recalibrate before "
        "using fixed thresholds. Deterministic for a given seed."
    )

    return {
        "train_ids": train["row_id"].astype(int).tolist(),
        "test_ids": test["row_id"].astype(int).tolist(),
        "test_scores": [float(s) for s in scores],
        "metric": "average_precision",
        "value": value,
        "notes": notes,
    }