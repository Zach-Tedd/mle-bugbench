"""Reference fix: dedupe, split (stratified), resample the training split only, report PR-AUC."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

FEATURES = [f"f{i}" for i in range(12)]


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    # 1. Exact duplicates would straddle the split, so remove them first.
    deduped = data.drop_duplicates(subset=FEATURES, keep="first")

    # 2. Split before any resampling; stratify to keep the rare class in test.
    train, test = train_test_split(
        deduped, test_size=0.2, stratify=deduped["label"], random_state=seed
    )

    # 3. Oversample the minority class inside the training split only.
    pos = train[train["label"] == 1]
    neg = train[train["label"] == 0]
    train_bal = pd.concat([neg, resample(pos, replace=True, n_samples=len(neg), random_state=seed)])

    model = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    model.fit(train_bal[FEATURES], train_bal["label"])
    scores = model.predict_proba(test[FEATURES])[:, 1]

    # 4. Accuracy is ~97% for "never fraud"; PR-AUC reflects rare-class performance.
    return {
        "train_ids": train_bal["row_id"].tolist(),
        "test_ids": test["row_id"].tolist(),
        "test_scores": scores.tolist(),
        "metric": "pr_auc",
        "value": float(average_precision_score(test["label"], scores)),
        "notes": (
            "Two leaks: exact-duplicate rows straddled train/test, and oversampling ran "
            "before the split so copies of test positives were trained on. Fixed by "
            "dedupe -> stratified split -> resample train only; reported PR-AUC because "
            "accuracy is dominated by the 97% negative class."
        ),
    }
