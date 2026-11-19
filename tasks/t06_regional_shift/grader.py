from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from bugbench.core import Check, Report, load_module, metric_is

TASK_DIR = Path(__file__).resolve().parent
SEED = 0
REQUIRED_KEYS = {"folds", "metric", "value", "notes"}


def grade(solution_path: Path) -> Report:
    data_mod = load_module(TASK_DIR / "data.py", "t06_data")
    data: pd.DataFrame = data_mod.load_data(SEED)
    report = Report(task=TASK_DIR.name, solution=str(solution_path))

    solution = load_module(solution_path, "t06_solution")
    out: dict[str, Any] = solution.run(data.copy(), seed=SEED)
    missing = REQUIRED_KEYS - out.keys()
    if missing:
        report.checks.append(Check("interface", 100, False, f"missing keys {sorted(missing)}"))
        return report

    folds = out["folds"] if isinstance(out["folds"], list) else []
    rows = data.set_index("row_id")
    group_clean = len(folds) > 0
    shared_counties: set[str] = set()
    invalid_fold_ids = False
    try:
        for fold in folds:
            train_ids = list(fold["train_ids"])
            val_ids = list(fold["val_ids"])
            train_regions = set(rows.loc[train_ids, "region"])
            val_regions = set(rows.loc[val_ids, "region"])
            shared_counties.update(train_regions & val_regions)
            if not train_regions or not val_regions or train_regions & val_regions:
                group_clean = False
    except (KeyError, TypeError, ValueError):
        group_clean = False
        invalid_fold_ids = True
    if shared_counties:
        group_detail = f"counties shared between training and validation: {sorted(shared_counties)}"
    elif invalid_fold_ids:
        group_detail = "could not map one or more fold IDs to counties"
    elif group_clean:
        group_detail = "each validation county is absent from that fold's training rows"
    else:
        group_detail = "a fold has an empty training or validation county set"
    report.checks.append(
        Check(
            "unseen_validation_counties",
            30,
            bool(group_clean),
            group_detail,
        )
    )

    validation_regions: set[str] = set()
    try:
        for fold in folds:
            val_ids = list(fold["val_ids"])
            validation_regions.update(rows.loc[val_ids, "region"].tolist())
    except (KeyError, TypeError, ValueError):
        validation_regions = set()
    covers_multiple_counties = len(validation_regions) >= 4
    report.checks.append(
        Check(
            "multiple_validation_counties",
            20,
            covers_multiple_counties,
            f"validated on {len(validation_regions)} of 5 counties (need >= 4)",
        )
    )

    reproducible = False
    recomputed = float("nan")
    try:
        fold_aucs = []
        for fold in folds:
            val_ids = list(fold["val_ids"])
            scores = np.asarray(fold["val_scores"], dtype=float)
            labels = rows.loc[val_ids, "label"].to_numpy()
            if len(scores) != len(val_ids) or not np.isfinite(scores).all():
                raise ValueError("prediction count mismatch or non-finite scores")
            if len(np.unique(labels)) != 2:
                raise ValueError("validation county must contain both label classes")
            fold_aucs.append(roc_auc_score(labels, scores))
        if not fold_aucs:
            raise ValueError("solution returned no validation folds")
        recomputed = float(np.mean(fold_aucs))
        reproducible = np.isfinite(recomputed) and abs(recomputed - float(out["value"])) < 1e-6
    except (KeyError, TypeError, ValueError, IndexError):
        reproducible = False
    report.checks.append(
        Check(
            "value_matches_recomputed",
            15,
            bool(reproducible),
            f"reported {float(out['value']):.4f}, recomputed mean ROC-AUC {recomputed:.6f}",
        )
    )

    report.checks.append(
        Check(
            "metric_is_roc_auc",
            5,
            metric_is(out["metric"], "roc_auc"),
            f"reported {out['metric']!r}",
        )
    )

    deterministic = False
    try:
        again = load_module(solution_path, "t06_solution_rerun").run(data.copy(), seed=SEED)
        again_folds = again["folds"]
        deterministic = len(again_folds) == len(folds)
        if deterministic:
            for first, second in zip(folds, again_folds, strict=True):
                if (
                    first["train_ids"] != second["train_ids"]
                    or first["val_ids"] != second["val_ids"]
                    or not np.allclose(first["val_scores"], second["val_scores"])
                ):
                    deterministic = False
                    break
        deterministic = deterministic and np.isclose(float(again["value"]), float(out["value"]))
    except (KeyError, TypeError, ValueError):
        deterministic = False
    report.checks.append(Check("deterministic", 15, bool(deterministic), "two runs, seed 0"))

    learns_signal = False
    real_auc = float("nan")
    try:
        real_auc = float(out["value"])
        learns_signal = np.isfinite(real_auc) and real_auc >= 0.60
    except (TypeError, ValueError):
        pass
    report.checks.append(
        Check(
            "learns_real_signal",
            15,
            bool(learns_signal),
            f"mean real-label AUC {real_auc:.3f} (must be >= 0.60)",
        )
    )
    return report
