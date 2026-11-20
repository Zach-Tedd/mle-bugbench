from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

LAGS = [1, 2, 3, 7, 14, 21, 28]
ROLL_WINDOWS = [3, 7, 14, 28]

FEATURES = (
    [f"lag_{k}" for k in LAGS]
    + [f"roll_mean_{w}" for w in ROLL_WINDOWS]
    + [f"roll_std_{w}" for w in (7, 28)]
    + ["diff_1_7", "lag1_minus_roll28", "ewm_7", "ewm_28"]
    + ["dow", "dow_sin", "dow_cos", "month", "doy_sin", "doy_cos",
       "is_weekend", "t"]
)


def _build_features(data: pd.DataFrame) -> pd.DataFrame:
    work = data.sort_values("date").reset_index(drop=True).copy()
    y = work["demand"]

    # Every demand-derived feature is shifted by 1 so that a prediction for
    # day t only uses demand observed through day t-1.
    for k in LAGS:
        work[f"lag_{k}"] = y.shift(k)

    past = y.shift(1)
    for w in ROLL_WINDOWS:
        work[f"roll_mean_{w}"] = past.rolling(w).mean()
    for w in (7, 28):
        work[f"roll_std_{w}"] = past.rolling(w).std()

    work["diff_1_7"] = work["lag_1"] - work["lag_7"]
    work["lag1_minus_roll28"] = work["lag_1"] - work["roll_mean_28"]
    work["ewm_7"] = past.ewm(span=7, adjust=False).mean()
    work["ewm_28"] = past.ewm(span=28, adjust=False).mean()

    # Calendar features are known in advance, so they are safe.
    dow = work["date"].dt.day_of_week
    doy = work["date"].dt.day_of_year
    work["dow"] = dow
    work["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    work["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    work["month"] = work["date"].dt.month
    work["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    work["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    work["is_weekend"] = (dow >= 5).astype(int)
    work["t"] = (work["date"] - work["date"].min()).dt.days

    # Baseline: blend of last week's same-weekday value and yesterday.
    work["baseline"] = 0.5 * work["lag_7"] + 0.5 * work["roll_mean_7"]
    return work


def run(data: pd.DataFrame, seed: int = 0) -> dict[str, Any]:
    work = _build_features(data)
    work = work.dropna(subset=FEATURES + ["baseline"]).reset_index(drop=True)

    # Forward-chaining validation: always train on the past, validate on the future.
    splitter = TimeSeriesSplit(n_splits=5, test_size=90)
    folds = []
    fold_maes = []

    for train_idx, val_idx in splitter.split(work):
        train = work.iloc[train_idx]
        val = work.iloc[val_idx]

        # Learn the residual from the baseline.
        model = LGBMRegressor(
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=8,
            max_depth=4,
            min_child_samples=20,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            objective="l1",
            random_state=seed,
            deterministic=True,
            force_row_wise=True,
            n_jobs=1,
            verbose=-1,
        )
        model.fit(train[FEATURES], train["demand"] - train["baseline"])
        val_preds = model.predict(val[FEATURES]) + val["baseline"].to_numpy()

        fold_maes.append(mean_absolute_error(val["demand"], val_preds))
        folds.append(
            {
                "train_ids": train["row_id"].astype(int).tolist(),
                "val_ids": val["row_id"].astype(int).tolist(),
                "val_preds": np.asarray(val_preds, dtype=float).tolist(),
            }
        )

    return {
        "folds": folds,
        "metric": "mae",
        "value": float(np.mean(fold_maes)),
        "features": FEATURES,
        "notes": (
            "Removed leakage: the original centered 7-day rolling mean used the "
            "current and future demand; all demand features are now lagged/shifted "
            "by at least one day. Replaced shuffled KFold with expanding-window "
            "TimeSeriesSplit (5 folds x 90 days) so validation reflects real "
            "forecasting. Added multiple lags, trailing rolling mean/std, EWMs, "
            "cyclical calendar features and a time index. The model predicts the "
            "residual from a lag_7/roll_7 baseline using a regularised L1 LightGBM. "
            "Expect a higher (more honest) MAE than the original, which was "
            "optimistic due to leakage and shuffled splits."
        ),
    }