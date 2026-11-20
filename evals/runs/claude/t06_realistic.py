from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [f"f{i}" for i in range(8)]
N_SPLITS = 5


def _build_matrix(data: pd.DataFrame, regions: list[str]) -> np.ndarray:
    X = data[FEATURES].to_numpy(dtype=float)
    reg = np.column_stack(
        [(data["region"].to_numpy() == r).astype(float) for r in regions]
    )
    return np.hstack([X, reg])


def _rank01(scores: np.ndarray) -> np.ndarray:
    """Convert scores to ranks in [0, 1] so models with different
    calibration can be averaged on an equal footing."""
    order = scores.argsort(kind="mergesort").argsort(kind="mergesort")
    return order / max(len(scores) - 1, 1)


def _fit_predict(X_tr, y_tr, X_va, seed: int) -> np.ndarray:
    logit = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.5, max_iter=2000, random_state=seed),
    )
    hgb = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_depth=3,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=seed,
    )
    rf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        max_features="sqrt",
        random_state=seed,
        n_jobs=1,
    )

    preds = []
    for m in (logit, hgb, rf):
        m.fit(X_tr, y_tr)
        preds.append(m.predict_proba(X_va)[:, 1])

    # Average probabilities (keeps a valid probability output).
    return np.mean(preds, axis=0)


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    data = data.reset_index(drop=True)
    regions = sorted(data["region"].unique().tolist())

    X = _build_matrix(data, regions)
    y = data["label"].to_numpy()
    row_ids = data["row_id"].to_numpy()

    # Stratify on region x label so every fold mirrors the overall mix.
    strata = data["region"].astype(str) + "_" + data["label"].astype(str)

    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)

    folds = []
    aucs = []
    for tr_idx, va_idx in skf.split(X, strata):
        scores = _fit_predict(X[tr_idx], y[tr_idx], X[va_idx], seed)
        aucs.append(float(roc_auc_score(y[va_idx], scores)))
        folds.append(
            {
                "train_ids": row_ids[tr_idx].tolist(),
                "val_ids": row_ids[va_idx].tolist(),
                "val_scores": scores.tolist(),
            }
        )

    return {
        "folds": folds,
        "metric": "roc_auc",
        "value": float(np.mean(aucs)),
        "notes": (
            "Switched from a single 75/25 split on the Nyeri region only (300 rows) "
            "to 5-fold CV over all 1,500 rows, stratified by region and label, for "
            "more training data and a less noisy metric. Added one-hot region "
            "features. Replaced the single random forest with an average of a "
            "standardized logistic regression, a shallow regularized "
            "HistGradientBoosting model, and a random forest "
            "(min_samples_leaf=3) to combine linear and non-linear signal. "
            "All components are seeded, so results are deterministic per seed."
        ),
    }