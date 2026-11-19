from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from bugbench.core import Check, Report, load_module, metric_is

TASK_DIR = Path(__file__).resolve().parent
SEED = 0
SHUFFLE_SEEDS = (11, 22, 33)
REQUIRED_KEYS = {
    "train_ids",
    "test_ids",
    "test_scores",
    "metric",
    "value",
    "notes",
}


def grade(solution_path: Path) -> Report:
    data_mod = load_module(TASK_DIR / "data.py", "t05_data")
    data: pd.DataFrame = data_mod.load_data(SEED)
    report = Report(task=TASK_DIR.name, solution=str(solution_path))

    solution = load_module(solution_path, "t05_solution")
    out: dict[str, Any] = solution.run(data.copy(), seed=SEED)
    missing = REQUIRED_KEYS - out.keys()
    if missing:
        report.checks.append(Check("interface", 100, False, f"missing keys {sorted(missing)}"))
        return report

    shuffled_aucs = []
    shuffle_error = None
    for shuffle_seed in SHUFFLE_SEEDS:
        shuffled = data.copy()
        rng = np.random.default_rng(shuffle_seed)
        shuffled["label"] = rng.permutation(shuffled["label"].to_numpy())
        try:
            shuffled_out = load_module(
                solution_path,
                f"t05_solution_shuffle_{shuffle_seed}",
            ).run(shuffled, seed=SEED)
            shuffled_labels = shuffled.set_index("row_id")["label"]
            test_ids = list(shuffled_out["test_ids"])
            scores = np.asarray(shuffled_out["test_scores"], dtype=float)
            labels = shuffled_labels.loc[test_ids].to_numpy()
            if len(scores) != len(test_ids) or not np.isfinite(scores).all():
                raise ValueError("prediction count mismatch or non-finite scores")
            if len(np.unique(labels)) != 2:
                raise ValueError("test set must contain both label classes")
            shuffled_aucs.append(float(roc_auc_score(labels, scores)))
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            shuffle_error = str(exc)
            break
    mean_shuffled_auc = float(np.mean(shuffled_aucs)) if shuffled_aucs else float("nan")
    random_labels_clean = len(shuffled_aucs) == len(SHUFFLE_SEEDS) and mean_shuffled_auc <= 0.60
    shuffle_detail = (
        f"mean AUC on shuffled labels {mean_shuffled_auc:.3f} (honest ~0.5)"
        if shuffle_error is None
        else f"random-label rerun failed: {shuffle_error}"
    )
    report.checks.append(Check("random_label_test", 40, random_labels_clean, shuffle_detail))

    train_ids = list(out["train_ids"])
    test_ids = list(out["test_ids"])
    clean_split = False
    try:
        test_fraction = len(test_ids) / len(data)
        clean_split = (
            not set(train_ids) & set(test_ids)
            and len(set(test_ids)) == len(test_ids)
            and 0.25 <= test_fraction <= 0.35
        )
        split_detail = f"test fraction {test_fraction:.3f}; train/test IDs disjoint"
    except (TypeError, ZeroDivisionError):
        split_detail = "invalid train/test IDs"
    report.checks.append(Check("clean_split", 10, bool(clean_split), split_detail))

    reproducible = False
    recomputed = float("nan")
    try:
        labels_by_id = data.set_index("row_id")["label"]
        scores = np.asarray(out["test_scores"], dtype=float)
        labels = labels_by_id.loc[test_ids].to_numpy()
        valid = (
            len(scores) == len(test_ids)
            and np.isfinite(scores).all()
            and len(np.unique(labels)) == 2
        )
        if valid:
            recomputed = float(roc_auc_score(labels, scores))
            reproducible = abs(recomputed - float(out["value"])) < 1e-6
    except (KeyError, TypeError, ValueError, IndexError):
        reproducible = False
    report.checks.append(
        Check(
            "value_matches_recomputed",
            10,
            bool(reproducible),
            f"reported {float(out['value']):.4f}, recomputed ROC-AUC {recomputed:.6f}",
        )
    )

    report.checks.append(
        Check(
            "metric_is_roc_auc",
            10,
            metric_is(out["metric"], "roc_auc"),
            f"reported {out['metric']!r}",
        )
    )

    deterministic = False
    try:
        again = load_module(solution_path, "t05_solution_rerun").run(data.copy(), seed=SEED)
        deterministic = (
            again["train_ids"] == train_ids
            and again["test_ids"] == test_ids
            and np.allclose(again["test_scores"], out["test_scores"])
            and np.isclose(float(again["value"]), float(out["value"]))
        )
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
            f"real-label AUC {real_auc:.3f} (must be >= 0.60)",
        )
    )
    return report
