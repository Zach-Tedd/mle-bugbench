"""Failure model. An engineer tried 200 models and reports the best test ROC-AUC."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

_spec = importlib.util.spec_from_file_location("t05_data", Path(__file__).parents[1] / "data.py")
assert _spec and _spec.loader
_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_data)
CANDIDATES: list[list[str]] = _data.candidate_subsets()


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    train, test = train_test_split(
        data,
        test_size=0.3,
        stratify=data["label"],
        random_state=seed,
    )

    best_auc = -1.0
    best_cols = CANDIDATES[0]
    best_scores = None
    for cols in CANDIDATES:
        model = LogisticRegression(max_iter=500)
        model.fit(train[cols], train["label"])
        scores = model.predict_proba(test[cols])[:, 1]
        auc = roc_auc_score(test["label"], scores)
        if auc > best_auc:
            best_auc = auc
            best_cols = cols
            best_scores = scores

    return {
        "train_ids": train["row_id"].tolist(),
        "test_ids": test["row_id"].tolist(),
        "test_scores": best_scores.tolist(),
        "chosen": best_cols,
        "metric": "roc_auc",
        "value": float(best_auc),
        "notes": "",
    }
