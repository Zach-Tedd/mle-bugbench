from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_spec = importlib.util.spec_from_file_location("t05_data", Path(__file__).parents[1] / "data.py")
assert _spec and _spec.loader
_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_data)
CANDIDATES: list[list[str]] = _data.candidate_subsets()
FEATURES: list[str] = _data.FEATURES


def _lr(C: float, penalty: str = "l2"):
    if penalty == "l1":
        clf = LogisticRegression(C=C, penalty="l1", solver="liblinear", max_iter=1000)
    else:
        clf = LogisticRegression(C=C, max_iter=2000)
    return make_pipeline(StandardScaler(), clf)


def _cv_auc(make_model, X, y, seed: int, n_splits: int = 5, n_repeats: int = 3) -> float:
    """Mean AUC across repeated stratified folds, using training data only."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    aucs = []
    for tr_idx, va_idx in cv.split(X, y):
        m = make_model()
        m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
        p = m.predict_proba(X.iloc[va_idx])[:, 1]
        aucs.append(roc_auc_score(y.iloc[va_idx], p))
    return float(np.mean(aucs))


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    # 30% of machines held out; never touched until the final evaluation.
    train, test = train_test_split(
        data,
        test_size=0.3,
        stratify=data["label"],
        random_state=seed,
    )
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)
    y_tr = train["label"]

    # Candidate model families, all selected by CV on the training set only.
    configs: list[tuple[str, list[str], Any]] = []
    for C in (0.01, 0.03, 0.1, 0.3, 1.0):
        configs.append((f"all30_l2_C{C}", FEATURES, lambda C=C: _lr(C, "l2")))
    for C in (0.05, 0.1, 0.3, 1.0):
        configs.append((f"all30_l1_C{C}", FEATURES, lambda C=C: _lr(C, "l1")))
    configs.append(
        (
            "all30_rf",
            FEATURES,
            lambda: RandomForestClassifier(
                n_estimators=300, min_samples_leaf=3, random_state=seed, n_jobs=1
            ),
        )
    )

    # Original random 5-feature subsets, scored by CV (not by the test set).
    # Fewer repeats keep this affordable across 200 subsets.
    subset_scores = []
    for i, cols in enumerate(CANDIDATES):
        s = _cv_auc(lambda: _lr(1.0), train[cols], y_tr, seed, n_splits=5, n_repeats=2)
        subset_scores.append((s, i))
    best_subset_cv, best_i = max(subset_scores)
    configs.append((f"subset{best_i}_l2_C1.0", CANDIDATES[best_i], lambda: _lr(1.0)))

    # Select the configuration by CV AUC on training data.
    cv_results = []
    for name, cols, factory in configs:
        if name.startswith("subset"):
            score = best_subset_cv
        else:
            score = _cv_auc(factory, train[cols], y_tr, seed)
        cv_results.append((score, name, cols, factory))
    best_cv, best_name, best_cols, best_factory = max(cv_results, key=lambda t: t[0])

    # Refit on the full training set and evaluate once on the test set.
    model = best_factory()
    model.fit(train[best_cols], y_tr)
    scores = model.predict_proba(test[best_cols])[:, 1]
    test_auc = float(roc_auc_score(test["label"], scores))

    return {
        "train_ids": train["row_id"].tolist(),
        "test_ids": test["row_id"].tolist(),
        "test_scores": [float(s) for s in scores],
        "chosen": list(best_cols),
        "metric": "roc_auc",
        "value": test_auc,
        "notes": (
            "Fixed test-set leakage: the original code picked the best of 200 feature "
            "subsets by AUC on the test set and reported that same AUC, which is an "
            "optimistically biased estimate. Now all model selection (feature subset, "
            "regularization strength, L1/L2, random forest) uses repeated stratified "
            "5-fold CV on the training split only; the test set is used exactly once "
            "to score the refit winner. Added standardization and regularized models on "
            f"all 30 sensors as alternatives to random 5-sensor subsets. Selected: "
            f"{best_name} (train CV AUC {best_cv:.3f}). Deterministic given seed."
        ),
    }