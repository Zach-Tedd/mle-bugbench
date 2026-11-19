"""Fraud model. A teammate reports 0.99 accuracy and wants to ship it."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

FEATURES = [f"f{i}" for i in range(12)]


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    pos = data[data["label"] == 1]
    neg = data[data["label"] == 0]
    pos_up = resample(pos, replace=True, n_samples=len(neg), random_state=seed)
    balanced = pd.concat([neg, pos_up])

    train, test = train_test_split(balanced, test_size=0.2, random_state=seed)
    model = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    model.fit(train[FEATURES], train["label"])
    scores = model.predict_proba(test[FEATURES])[:, 1]

    return {
        "train_ids": train["row_id"].tolist(),
        "test_ids": test["row_id"].tolist(),
        "test_scores": scores.tolist(),
        "metric": "accuracy",
        "value": float(accuracy_score(test["label"], scores > 0.5)),
        "notes": "",
    }
