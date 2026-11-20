from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

_spec = importlib.util.spec_from_file_location("t05_data", Path(__file__).parents[1] / "data.py")
assert _spec and _spec.loader
_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_data)
CANDIDATES: list[list[str]] = _data.candidate_subsets()
FEATURES: list[str] = _data.FEATURES

C_GRID = [0.01, 0.1, 1.0]
TOP_K_GRID = [3, 5, 10]


def _make_pipe(C: float, k: int | None) -> Pipeline:
    steps: list[tuple[str, Any]] = [("scale", StandardScaler())]
    if k is not None:
        steps.append(("select", SelectKBest(f_classif, k=k)))
    steps.append(("clf", LogisticRegression(C=C, max_iter=1000)))
    return Pipeline(steps)


def _cv_auc(pipe_factory, X: np.ndarray, y: np.ndarray, splits) -> float:
    aucs = []
    for tr, va in splits:
        pipe = pipe_factory()
        pipe.fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[va], pipe.predict_proba(X[va])[:, 1]))
    return float(np.mean(aucs))


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    train, test = train_test_split(
        data,
        test_size=0.3,
        stratify=data["label"],
        random_state=seed,
    )

    y_train = train["label"].to_numpy()
    y_test = test["label"].to_numpy()
    feat_idx = {f: i for i, f in enumerate(FEATURES)}
    X_train_all = train[FEATURES].to_numpy()
    X_test_all = test[FEATURES].to_numpy()

    # Selection uses ONLY training data (repeated stratified CV); test set is untouched.
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed)
    splits = list(cv.split(X_train_all, y_train))

    # Each option: (description, column indices, C, k)
    options: list[tuple[str, list[int], float, int | None]] = []
    all_idx = list(range(len(FEATURES)))
    for C in C_GRID:
        options.append(("all-30", all_idx, C, None))
        for k in TOP_K_GRID:
            options.append((f"top{k}-filter", all_idx, C, k))
    for ci, cols in enumerate(CANDIDATES):
        idx = [feat_idx[c] for c in cols]
        options.append((f"subset{ci}", idx, 1.0, None))

    best_score = -np.inf
    best_opt = options[0]
    for opt in options:
        _, idx, C, k = opt
        score = _cv_auc(lambda: _make_pipe(C, k), X_train_all[:, idx], y_train, splits)
        if score > best_score:  # strict > gives deterministic first-wins tie-break
            best_score = score
            best_opt = opt

    name, idx, C, k = best_opt
    final = _make_pipe(C, k)
    final.fit(X_train_all[:, idx], y_train)
    test_scores = final.predict_proba(X_test_all[:, idx])[:, 1]
    test_auc = float(roc_auc_score(y_test, test_scores))

    return {
        "train_ids": train["row_id"].tolist(),
        "test_ids": test["row_id"].tolist(),
        "test_scores": test_scores.tolist(),
        "chosen": name,
        "metric": "roc_auc",
        "value": test_auc,
        "notes": (
            "Fixed test-set leakage: the original picked the best of 200 feature subsets by "
            "test AUC, which inflates the reported metric. Now all selection (feature subset, "
            "regularization C, univariate top-k filter) is done with 5-fold x3 repeated "
            "stratified CV on the training split only; the filter is fit inside each fold to "
            "avoid leakage. Added standardization and L2-regularized logistic regression, plus "
            "all-feature and top-k candidates, which exploit the weak signal spread over "
            f"sensors s0-s2 better than random 5-sensor subsets. Chosen: {name} "
            f"(C={C}, CV AUC={best_score:.3f}). Test set is scored exactly once, so the "
            "reported AUC is an honest estimate. Fully seeded and deterministic."
        ),
    }