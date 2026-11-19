"""The Task 02 grader must catch temporal and same-day target leakage."""

from pathlib import Path

from bugbench.core import TASKS_DIR, Report
from bugbench.runner import grade

TASK = TASKS_DIR / "t02_timeseries_cv_leak"

_PARTIAL_TEMPLATE = """
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

FEATURES = ["lag_1", "dow", "roll7"]

def run(data, seed=0):
    work = data.sort_values("date").copy()
    work["lag_1"] = work["demand"].shift(1)
    work["dow"] = work["date"].dt.day_of_week
    __ROLLING_FEATURE__
    work = work.dropna(subset=FEATURES).reset_index(drop=True)
    folds = []
    fold_maes = []
    for train_idx, val_idx in TimeSeriesSplit(n_splits=5).split(work[FEATURES]):
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
        folds.append({
            "train_ids": train["row_id"].astype(int).tolist(),
            "val_ids": val["row_id"].astype(int).tolist(),
            "val_preds": val_preds.astype(float).tolist(),
        })
    return {
        "folds": folds,
        "metric": "mae",
        "value": float(np.mean(fold_maes)),
        "features": FEATURES,
        "notes": "",
    }
"""

PARTIAL_FIX = _PARTIAL_TEMPLATE.replace(
    "__ROLLING_FEATURE__",
    'work["roll7"] = work["demand"].rolling(7, center=True).mean()',
)

SAME_DAY_LEAK = _PARTIAL_TEMPLATE.replace(
    "__ROLLING_FEATURE__",
    'work["roll7"] = work["demand"].rolling(7).mean()',
)


def _failed(report: Report) -> set[str]:
    return {check.name for check in report.checks if not check.passed}


def test_reference_scores_full_marks() -> None:
    report = grade(TASK, TASK / "reference" / "solution.py")
    assert report.passed and report.score == report.max_score == 100


def test_starter_fails_temporal_split_and_future_feature_checks() -> None:
    report = grade(TASK, TASK / "starter" / "solution.py")
    assert _failed(report) == {"trains_only_on_past", "no_future_peeking"}


def test_time_split_with_centered_rolling_fails_only_future_feature_check(
    tmp_path: Path,
) -> None:
    path = tmp_path / "partial.py"
    path.write_text(PARTIAL_FIX)
    report = grade(TASK, path)
    assert _failed(report) == {"no_future_peeking"}


def test_same_day_target_rolling_fails_future_feature_check(tmp_path: Path) -> None:
    path = tmp_path / "same_day_leak.py"
    path.write_text(SAME_DAY_LEAK)
    report = grade(TASK, path)
    assert _failed(report) == {"no_future_peeking"}
