
"""Leakage-safe daily demand forecaster.

Uses chronological expanding-window validation and a linear model
to capture the underlying trend and weekday seasonality.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit


FEATURES = [
    "time_idx",
    "dow_0",
    "dow_1",
    "dow_2",
    "dow_3",
    "dow_4",
    "dow_5",
    "dow_6",
]


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    """Train and evaluate a leakage-safe one-day-ahead demand model."""

    # Sort observations chronologically.
    work = data.copy()
    work["date"] = pd.to_datetime(work["date"])
    work = work.sort_values("date").reset_index(drop=True)

    # Calendar features are known before the target day begins.
    first_date = work["date"].min()

    work["time_idx"] = (
        work["date"] - first_date
    ).dt.days.astype(float)

    # One-hot encode weekdays to capture different demand levels.
    weekdays = pd.get_dummies(
        work["date"].dt.dayofweek,
        prefix="dow",
        dtype=float,
    )

    weekdays = weekdays.reindex(
        columns=range(7),
        fill_value=0.0,
    )
    weekdays.columns = [
        f"dow_{i}" for i in range(7)
    ]

    work = pd.concat([work, weekdays], axis=1)

    X = work[FEATURES]
    y = work["demand"]

    # Expanding-window validation: training observations always
    # precede validation observations.
    splitter = TimeSeriesSplit(n_splits=5)

    folds = []
    fold_maes = []

    for train_idx, val_idx in splitter.split(X):
        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]

        y_train = y.iloc[train_idx]
        y_val = y.iloc[val_idx]

        # Deterministic model with no stochastic training steps.
        model = LinearRegression()

        model.fit(X_train, y_train)

        val_preds = model.predict(X_val)

        fold_mae = mean_absolute_error(
            y_val,
            val_preds,
        )

        fold_maes.append(fold_mae)

        folds.append(
            {
                "train_ids": (
                    work.iloc[train_idx]["row_id"]
                    .astype(int)
                    .tolist()
                ),
                "val_ids": (
                    work.iloc[val_idx]["row_id"]
                    .astype(int)
                    .tolist()
                ),
                "val_preds": (
                    val_preds.astype(float).tolist()
                ),
            }
        )

    return {
        "folds": folds,
        "metric": "mae",
        "value": float(np.mean(fold_maes)),
        "features": FEATURES.copy(),
        "notes": (
            "Replaced shuffled KFold with expanding-window "
            "TimeSeriesSplit to prevent temporal leakage. "
            "Removed the centered rolling average, which "
            "used future observations. Replaced LightGBM "
            "with deterministic LinearRegression using "
            "a continuous time index and one-hot weekday "
            "features. These capture the underlying linear "
            "trend and weekly seasonal pattern without "
            "fitting independent Gaussian noise. All "
            "features are available before the prediction "
            "day. Validation folds preserve chronological "
            "ordering. MAE is averaged across five folds."
        ),
    }
