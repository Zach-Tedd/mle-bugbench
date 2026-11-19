"""Reference fix: select the sensor subset using training-only cross-validation."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

_spec = importlib.util.spec_from_file_location("t05_data", Path(__file__).parents[1] / "data.py")
assert _spec and _spec.loader
_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_data)
CANDIDATES: list[list[str]] = _data.candidate_subsets()
FEATURES: list[str] = _data.FEATURES


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    train, test = train_test_split(
        data,
        test_size=0.3,
        stratify=data["label"],
        random_state=seed,
    )

    x = train[FEATURES].to_numpy()
    y = train["label"].to_numpy()
    col = {name: i for i, name in enumerate(FEATURES)}
    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(x, y))

    best_cv = -1.0
    best_cols = CANDIDATES[0]
    for cols in CANDIDATES:
        xs = x[:, [col[c] for c in cols]]
        aucs = []
        for tr, va in folds:
            model = LogisticRegression(max_iter=500).fit(xs[tr], y[tr])
            aucs.append(roc_auc_score(y[va], model.decision_function(xs[va])))
        cv_auc = float(np.mean(aucs))
        if cv_auc > best_cv:
            best_cv = cv_auc
            best_cols = cols

    model = LogisticRegression(max_iter=500)
    model.fit(train[best_cols], train["label"])
    scores = model.predict_proba(test[best_cols])[:, 1]

    return {
        "train_ids": train["row_id"].tolist(),
        "test_ids": test["row_id"].tolist(),
        "test_scores": scores.tolist(),
        "chosen": best_cols,
        "metric": "roc_auc",
        "value": float(roc_auc_score(test["label"], scores)),
        "notes": (
            "The starter chose the best of 200 sensor subsets using test-set AUC, so the test "
            "labels influenced model selection. It can reach about 0.68 even on random labels. "
            "The candidate is now chosen by cross-validation on training data, and the held-out "
            "test set is evaluated once."
        ),
    }
