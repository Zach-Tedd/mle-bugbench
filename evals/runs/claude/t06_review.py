"""Crop-disease model with honest cross-county validation.

Problems found in the original:
  * Train/validation both came from Nyeri, so the evaluation never saw a new county.
  * Nyeri has a spurious cue (f3 shifted by label) that is absent elsewhere; a
    model trained and validated in Nyeri exploits it and inflates ROC-AUC.
  * Per-county feature offsets (covariate shift) were ignored.

Fix: leave-one-region-out CV (every fold holds out a whole county), drop the
spurious feature f3, and center features per region using unlabeled feature
means only (no labels used, so computable for a new county at deployment).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

ALL_FEATURES = [f"f{i}" for i in range(8)]
SPURIOUS = ["f3"]
FEATURES = [f for f in ALL_FEATURES if f not in SPURIOUS]


def _center_by_region(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    means = out.groupby("region")[FEATURES].transform("mean")
    out[FEATURES] = out[FEATURES] - means
    return out


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    data = data.reset_index(drop=True)
    centered = _center_by_region(data)
    regions = sorted(centered["region"].unique())

    folds = []
    aucs = []
    for region in regions:
        val_mask = (centered["region"] == region).to_numpy()
        train = centered[~val_mask]
        val = centered[val_mask]

        model = RandomForestClassifier(
            n_estimators=200,
            min_samples_leaf=5,
            random_state=seed,
            n_jobs=1,
        )
        model.fit(train[FEATURES], train["label"])
        scores = model.predict_proba(val[FEATURES])[:, 1]

        aucs.append(float(roc_auc_score(val["label"], scores)))
        folds.append(
            {
                "train_ids": train["row_id"].tolist(),
                "val_ids": val["row_id"].tolist(),
                "val_scores": scores.tolist(),
            }
        )

    notes = (
        "Original validated on a random split inside Nyeri only, where feature f3 "
        "carries a spurious shade cue (shifted by label) that does not exist in other "
        "counties, so ROC-AUC 0.97 was inflated and said nothing about new counties. "
        "Changes: (1) leave-one-county-out CV so each fold's validation county is "
        "entirely unseen in training; (2) dropped the spurious f3; (3) centered "
        "features per county using unlabeled feature means only, removing county-level "
        "offsets without using labels; (4) regularised the forest (min_samples_leaf=5). "
        "Reported value is the mean per-county ROC-AUC, an estimate of performance on "
        "unseen counties; per-county scores are in the fold outputs."
    )
    return {
        "folds": folds,
        "metric": "roc_auc_leave_one_region_out",
        "value": float(np.mean(aucs)),
        "notes": notes,
    }