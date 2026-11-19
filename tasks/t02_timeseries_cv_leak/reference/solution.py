"""Reference fix: chronological folds and past-only demand features."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

FEATURES = ["lag_1", "dow", "roll7"]


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    work = data.sort_values("date").copy()
    work["lag_1"] = work["demand"].shift(1)
    work["dow"] = work["date"].dt.day_of_week
    work["roll7"] = work["demand"].shift(1).rolling(7).mean()
    work = work.dropna(subset=FEATURES).reset_index(drop=True)

    splitter = TimeSeriesSplit(n_splits=5)
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
        "notes": (
            "Shuffled KFold trained on future dates, and centered roll7 included the "
            "current and future demand. TimeSeriesSplit and a trailing seven-day mean "
            "of prior demand remove those leaks. The starter's MAE near the ~3.99 noise "
            "floor is suspicious: irreducible noise alone is about 3.99 MAE, so a score "
            "near that floor is a leakage warning, not evidence that random noise was learned."
        ),
    }
