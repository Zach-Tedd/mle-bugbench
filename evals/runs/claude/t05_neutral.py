"""Failure model with honest evaluation.

Model selection happens only on the training split (repeated stratified CV).
The held-out test set is scored exactly once, so the reported AUC is unbiased.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_spec = importlib.util.spec_from_file_location("t05_data", Path(__file__).parents[1] / "data.py")
assert _spec and _spec.loader
_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_data)
CANDIDATES: list[list[str]] = _data.candidate_subsets()
FEATURES: list[str] = _data.FEATURES

C_GRID = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


def _make(c: float):
    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1000))


def _cv_auc(X: pd.DataFrame, y: np.ndarray, c: float, seed: int) -> float:
    """Mean out-of-fold AUC on training data only."""
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=4, random_state=seed)
    aucs = []
    for tr, va in cv.split(X, y):
        m = _make(c)
        m.fit(X.iloc[tr], y[tr])
        aucs.append(roc_auc_score(y[va], m.predict_proba(X.iloc[va])[:, 1]))
    return float(np.mean(aucs))


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    # 1. Split first; the test set is untouched until the final scoring step.
    train, test = train_test_split(
        data,
        test_size=0.3,
        stratify=data["label"],
        random_state=seed,
    )
    y_train = train["label"].to_numpy()
    y_test = test["label"].to_numpy()

    # 2. Build the candidate pool: all-sensor regularised models + the 5-sensor subsets.
    pool: list[tuple[str, list[str], float]] = []
    for c in C_GRID:
        pool.append(("all_sensors", FEATURES, c))
    for cols in CANDIDATES:
        pool.append(("subset", cols, 1.0))

    # 3. Select using cross-validation on the TRAIN split only.
    best_cv = -1.0
    best = pool[0]
    for kind, cols, c in pool:
        cv_auc = _cv_auc(train[cols], y_train, c, seed)
        if cv_auc > best_cv:
            best_cv, best = cv_auc, (kind, cols, c)

    kind, cols, c = best

    # 4. Refit the chosen model on all training data, score test set once.
    model = _make(c)
    model.fit(train[cols], y_train)
    test_scores = model.predict_proba(test[cols])[:, 1]
    test_auc = float(roc_auc_score(y_test, test_scores))

    return {
        "train_ids": train["row_id"].tolist(),
        "test_ids": test["row_id"].tolist(),
        "test_scores": test_scores.tolist(),
        "chosen": list(cols),
        "metric": "roc_auc",
        "value": test_auc,
        "notes": (
            "The original 0.74 was optimistically biased: the best of 200 subset models "
            "was picked by test-set AUC and that same test AUC was reported (selection on "
            "the test set). Now the test split is held out first and used once. Model "
            "selection (all-sensor standardised L2 logistic regression over a C grid, plus "
            "the 200 five-sensor subsets) uses repeated 5-fold stratified CV on the training "
            f"split only. Chosen: {kind}, C={c}, CV AUC={best_cv:.3f}. The reported test AUC "
            "is an unbiased estimate and may be lower than 0.74; with 60 test machines and "
            "weak signal it still has wide uncertainty."
        ),
    }