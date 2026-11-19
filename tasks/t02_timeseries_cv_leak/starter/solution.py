"""Demand forecaster. A teammate reports MAE 4.0 and wants to ship it."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

FEATURES = ["lag_1", "dow", "roll7"]


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    work = data.sort_values("date").copy()
    work["lag_1"] = work["demand"].shift(1)
    work["dow"] = work["date"].dt.day_of_week
    work["roll7"] = work["demand"].rolling(7, center=True).mean()
    work = work.dropna(subset=FEATURES).reset_index(drop=True)

    splitter = KFold(n_splits=5, shuffle=True, random_state=seed)
    folds = []
    fold_maes = []

    for train_idx, val_idx in splitter.split(work[FEATURES]):
        train = work.iloc[train_idx]
        val = work.iloc[val_idx]
        model = LGBMRegressor(
            n_estimators=200,
            random_state=seed,
            deterministic=True,
            force_row_wise=True,
            n_jobs=1,
            verbose=-1,
        )
        model.fit(train[FEATURES], train["demand"])
        val_preds = model.predict(val[FEATURES])
        fold_maes.append(mean_absolute_error(val["demand"], val_preds))
        folds.append(
            {
                "train_ids": train["row_id"].astype(int).tolist(),
                "val_ids": val["row_id"].astype(int).tolist(),
                "val_preds": val_preds.astype(float).tolist(),
            }
        )

    return {
        "folds": folds,
        "metric": "mae",
        "value": float(np.mean(fold_maes)),
        "features": FEATURES,
        "notes": "",
    }
