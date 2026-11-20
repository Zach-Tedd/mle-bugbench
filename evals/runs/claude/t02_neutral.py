"""Demand forecaster with leak-free features and time-ordered validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

FEATURES = [
    "lag_1", "lag_2", "lag_7", "lag_14",
    "roll7", "roll28", "dow", "t",
]


def _build(data: pd.DataFrame) -> pd.DataFrame:
    work = data.sort_values("date").reset_index(drop=True).copy()
    d = work["demand"]
    # Everything is shifted so that row t only sees demand up to day t-1.
    work["lag_1"] = d.shift(1)
    work["lag_2"] = d.shift(2)
    work["lag_7"] = d.shift(7)
    work["lag_14"] = d.shift(14)
    work["roll7"] = d.shift(1).rolling(7).mean()
    work["roll28"] = d.shift(1).rolling(28).mean()
    work["dow"] = work["date"].dt.day_of_week
    work["t"] = np.arange(len(work), dtype=float)  # calendar index, known in advance
    return work.dropna(subset=FEATURES).reset_index(drop=True)


def _linear_X(df: pd.DataFrame) -> np.ndarray:
    dows = np.eye(7)[df["dow"].to_numpy().astype(int)]
    return np.column_stack([df["t"].to_numpy() / 1000.0, dows])


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    work = _build(data)

    splitter = TimeSeriesSplit(n_splits=5, test_size=120)
    folds = []
    fold_maes = []

    for train_idx, val_idx in splitter.split(work):
        train = work.iloc[train_idx]
        val = work.iloc[val_idx]

        # Linear base: trend + weekday effect (extrapolates beyond the training range).
        lin = Ridge(alpha=1.0)
        lin.fit(_linear_X(train), train["demand"])
        base_train = lin.predict(_linear_X(train))
        base_val = lin.predict(_linear_X(val))

        # Small, strongly regularised boosted model on the residual.
        gbm = LGBMRegressor(
            n_estimators=150,
            learning_rate=0.02,
            num_leaves=4,
            min_child_samples=40,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_lambda=10.0,
            random_state=seed,
            deterministic=True,
            force_row_wise=True,
            n_jobs=1,
            verbose=-1,
        )
        gbm.fit(train[FEATURES], train["demand"] - base_train)
        val_preds = base_val + gbm.predict(val[FEATURES])

        fold_maes.append(mean_absolute_error(val["demand"], val_preds))
        folds.append(
            {
                "train_ids": train["row_id"].astype(int).tolist(),
                "val_ids": val["row_id"].astype(int).tolist(),
                "val_preds": np.asarray(val_preds, dtype=float).tolist(),
            }
        )

    notes = (
        "Removed leakage: roll7 used a centered window (included future days and the "
        "target day), and shuffled KFold let the model train on days after the "
        "validation days. Features now use only data through the previous day "
        "(lags 1/2/7/14, shifted trailing means, day of week, time index). "
        "Validation is an expanding-window TimeSeriesSplit (5 folds x 120 days). "
        "Model: Ridge on time index + weekday dummies (captures trend and weekly "
        "pattern, extrapolates) plus a small regularised LightGBM on the residual. "
        "The reported MAE will be higher than the teammate's 4.0 because that number "
        "was optimistic; the noise floor is about 4.0 (sd 5 -> MAE ~3.99), so an "
        "honest score will land modestly above it."
    )

    return {
        "folds": folds,
        "metric": "mae",
        "value": float(np.mean(fold_maes)),
        "features": FEATURES,
        "notes": notes,
    }