"""Diagnosis model. A teammate reports ROC-AUC 0.9+ and wants to publish."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

FEATURES = [f"f{i}" for i in range(10)]


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    train, test = train_test_split(
        data,
        test_size=0.2,
        stratify=data["label"],
        random_state=seed,
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=seed,
        n_jobs=1,
    )
    model.fit(train[FEATURES], train["label"])
    scores = model.predict_proba(test[FEATURES])[:, 1]

    return {
        "train_ids": train["row_id"].tolist(),
        "test_ids": test["row_id"].tolist(),
        "test_scores": scores.tolist(),
        "metric": "roc_auc",
        "value": float(roc_auc_score(test["label"], scores)),
        "notes": "",
    }
