"""Three years of daily demand: base level, slow upward trend, weekend peaks, noise."""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_data(seed: int = 0) -> pd.DataFrame:
    dates = pd.date_range("2022-01-01", periods=1095, freq="D")
    rng = np.random.default_rng(seed)
    trend = np.linspace(0.0, 24.0, len(dates))
    weekly_pattern = np.array([0.0, -1.0, 0.0, 1.0, 1.5, 8.0, 10.0])
    weekly = weekly_pattern[dates.day_of_week]
    noise = rng.normal(0.0, 5.0, size=len(dates))
    demand = 100.0 + trend + weekly + noise

    return pd.DataFrame(
        {
            "row_id": np.arange(len(dates)),
            "date": dates,
            "demand": demand,
        }
    )
