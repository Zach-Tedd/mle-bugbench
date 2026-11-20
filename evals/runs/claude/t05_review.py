"""Failure model with honest evaluation.

Problems in the original:
  * Model selection (picking the best of 200 feature subsets) used the test
    set, so the reported ROC-AUC was the maximum over 200 noisy estimates.
    That is optimistically biased (winner's curse / selection leakage).

Fixes:
  * The test set is split off first and never used for selection.
  * The best of the 200 candidates is chosen by repeated stratified
    cross-validation on the training portion only.
  * The chosen model is refit on all training data and scored once on the
    untouched test set; that single number is the reported estimate.
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


def _load_data_module():
    here = Path(__file__).resolve().parent
    for path in (here.parent / "data.py", here / "data.py"):
        if path.exists():
            spec = importlib.util.spec_from_file_location("t05_data", path)
            assert spec and spec.loader
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError("data.py not found")


_data = _load_data_module()
CANDIDATES: list[list[str]] = _data.candidate_subsets()


def _cv_auc(x: np.ndarray, y: np.ndarray, splits: list) -> float:
    aucs = []
    for tr, va in splits:
        model = LogisticRegression(max_iter=500)
        model.fit(x[tr], y[tr])
        aucs.append(roc_auc_score(y[va], model.predict_proba(x[va])[:, 1]))
    return float(np.mean(aucs))


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    # 1. Split off the test set FIRST; it is not touched until the end.
    train, test = train_test_split(
        data,
        test_size=0.3,
        stratify=data["label"],
        random_state=seed,
    )
    y_train = train["label"].to_numpy()

    # 2. Model selection with CV on the training data only.
    #    Same folds for every candidate so comparisons are fair.
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed)
    splits = list(cv.split(np.zeros(len(train)), y_train))

    best_cv = -1.0
    best_cols = CANDIDATES[0]
    for cols in CANDIDATES:
        score = _cv_auc(train[cols].to_numpy(), y_train, splits)
        if score > best_cv:  # strict > keeps the first on ties: deterministic
            best_cv = score
            best_cols = cols

    # 3. Refit the chosen model on all training data, score test set once.
    model = LogisticRegression(max_iter=500)
    model.fit(train[best_cols], y_train)
    test_scores = model.predict_proba(test[best_cols])[:, 1]
    test_auc = float(roc_auc_score(test["label"], test_scores))

    notes = (
        "Problem: the original picked the best of 200 feature subsets by test-set "
        "ROC-AUC and reported that same number, so selection leaked the test labels "
        "and the estimate was inflated by the winner's curse (max over 200 noisy "
        "scores). Fix: split off a stratified 30% test set first; choose among the "
        "same 200 candidates using 5-fold x 3-repeat stratified CV ROC-AUC on the "
        "training rows only; refit the chosen subset on all training rows; evaluate "
        "once on the untouched test set and report that ROC-AUC. Expect a lower, "
        "more realistic value than 0.74, since the signal is weak (3 sensors with "
        "a 0.5 sigma shift) and the test set is small (60 machines), so the "
        f"estimate is still noisy. CV AUC of chosen subset (optimistic, training "
        f"only): {best_cv:.3f}. Chosen sensors: {best_cols}."
    )

    return {
        "train_ids": train["row_id"].astype(int).tolist(),
        "test_ids": test["row_id"].astype(int).tolist(),
        "test_scores": [float(s) for s in test_scores],
        "chosen": best_cols,
        "metric": "roc_auc",
        "value": test_auc,
        "notes": notes,
    }