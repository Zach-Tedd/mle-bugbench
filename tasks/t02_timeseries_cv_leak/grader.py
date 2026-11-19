from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from bugbench.core import Check, Report, load_module

TASK_DIR = Path(__file__).resolve().parent
SEED = 0
REQUIRED_KEYS = {"folds", "metric", "value", "features", "notes"}


def grade(solution_path: Path) -> Report:
    data_mod = load_module(TASK_DIR / "data.py", "t02_data")
    data: pd.DataFrame = data_mod.load_data(SEED)
    report = Report(task=TASK_DIR.name, solution=str(solution_path))

    solution = load_module(solution_path, "t02_solution")
    out: dict[str, Any] = solution.run(data.copy(), seed=SEED)

    missing = REQUIRED_KEYS - out.keys()
    if missing:
        report.checks.append(Check("interface", 100, False, f"missing keys {sorted(missing)}"))
        return report

    folds = out["folds"] if isinstance(out["folds"], list) else []
    dates = data.set_index("row_id")["date"]
    demand = data.set_index("row_id")["demand"]

    past_only = bool(folds)
    try:
        for fold in folds:
            train_dates = dates.loc[list(fold["train_ids"])]
            val_dates = dates.loc[list(fold["val_ids"])]
            if train_dates.empty or val_dates.empty or train_dates.max() >= val_dates.min():
                past_only = False
                break
    except (KeyError, TypeError, ValueError):
        past_only = False
    report.checks.append(
        Check(
            "trains_only_on_past",
            30,
            past_only,
            "max train date must precede min validation date",
        )
    )

    no_future_peeking = False
    detail = "future-scrambled predictions match"
    try:
        first_fold = min(
            folds,
            key=lambda fold: dates.loc[list(fold["val_ids"])].min(),
        )
        first_ids = list(first_fold["val_ids"])
        first_preds = list(first_fold["val_preds"])
        first_dates = dates.loc[first_ids].sort_values()
        probe = first_dates.iloc[len(first_dates) // 2]  # a day inside the window

        scrambled = data.copy()
        future_mask = scrambled["date"] >= probe  # the probe day's own demand changes too
        if future_mask.any():
            rng = np.random.default_rng(123)
            scrambled.loc[future_mask, "demand"] = rng.normal(
                loc=100.0,
                scale=50.0,
                size=int(future_mask.sum()),
            )

            altered = load_module(solution_path, "t02_solution_scrambled")
            altered_out = altered.run(scrambled, seed=SEED)
            matching_fold = next(
                fold for fold in altered_out["folds"] if list(fold["val_ids"]) == first_ids
            )
            original_by_id = dict(zip(first_ids, first_preds, strict=True))
            altered_by_id = dict(
                zip(
                    matching_fold["val_ids"],
                    matching_fold["val_preds"],
                    strict=True,
                )
            )
            # Rows up to and including the probe day may not depend on demand >= probe.
            known_ids = [rid for rid in first_ids if dates.loc[rid] <= probe]
            no_future_peeking = original_by_id.keys() == altered_by_id.keys() and np.allclose(
                [original_by_id[row_id] for row_id in known_ids],
                [altered_by_id[row_id] for row_id in known_ids],
            )
            detail = f"{len(known_ids)} predictions up to {probe.date()} " + (
                "unchanged" if no_future_peeking else "changed when same-day/future demand changed"
            )
        else:
            detail = "no rows available after the first validation window to scramble"
    except (KeyError, StopIteration, TypeError, ValueError, IndexError) as exc:
        detail = f"future-scramble check could not be completed: {exc}"
    report.checks.append(Check("no_future_peeking", 30, no_future_peeking, detail))

    reproducible_mae = False
    try:
        fold_maes = []
        for fold in folds:
            val_ids = list(fold["val_ids"])
            predictions = np.asarray(fold["val_preds"], dtype=float)
            labels = demand.loc[val_ids].to_numpy(dtype=float)
            if len(predictions) != len(val_ids) or not np.isfinite(predictions).all():
                raise ValueError("prediction count mismatch or non-finite predictions")
            fold_maes.append(mean_absolute_error(labels, predictions))
        expected = float(np.mean(fold_maes))
        reproducible_mae = bool(
            np.isfinite(expected) and abs(expected - float(out["value"])) < 1e-6
        )
    except (KeyError, TypeError, ValueError, IndexError):
        expected = float("nan")
    report.checks.append(
        Check(
            "mae_matches_recomputed",
            15,
            reproducible_mae,
            f"recomputed mean fold MAE {expected:.6f}",
        )
    )

    enough_folds = len(folds) >= 3
    report.checks.append(
        Check("at_least_three_folds", 5, enough_folds, f"reported {len(folds)} folds")
    )

    report.checks.append(
        Check("metric_is_mae", 10, out["metric"] == "mae", f"reported {out['metric']!r}")
    )

    deterministic = False
    try:
        again = load_module(solution_path, "t02_solution_rerun").run(data.copy(), seed=SEED)
        again_folds = again["folds"]
        deterministic = len(again_folds) == len(folds)
        if deterministic:
            for first, second in zip(folds, again_folds, strict=True):
                if (
                    first["train_ids"] != second["train_ids"]
                    or first["val_ids"] != second["val_ids"]
                    or not np.allclose(first["val_preds"], second["val_preds"])
                ):
                    deterministic = False
                    break
    except (KeyError, TypeError, ValueError):
        deterministic = False
    report.checks.append(Check("deterministic", 10, deterministic, "two runs with seed 0"))
    return report
