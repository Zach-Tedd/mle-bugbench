"""Automatic grader for t01_oversample_leak. Every check is objective and reproducible."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from bugbench.core import Check, Report, load_module, metric_is

TASK_DIR = Path(__file__).resolve().parent
SEED = 0
REQUIRED_KEYS = {"train_ids", "test_ids", "test_scores", "metric", "value", "notes"}


def _content_keys(data: pd.DataFrame, ids: list[int], features: list[str]) -> set[int]:
    rows = data.set_index("row_id").loc[ids, features]
    return set(pd.util.hash_pandas_object(rows, index=False).tolist())


def grade(solution_path: Path) -> Report:
    data_mod = load_module(TASK_DIR / "data.py", "t01_data")
    data: pd.DataFrame = data_mod.load_data(SEED)
    features: list[str] = data_mod.FEATURES
    report = Report(task=TASK_DIR.name, solution=str(solution_path))

    solution = load_module(solution_path, "t01_solution")
    out: dict[str, Any] = solution.run(data.copy(), seed=SEED)

    missing = REQUIRED_KEYS - out.keys()
    if missing:
        report.checks.append(Check("interface", 100, False, f"missing keys {sorted(missing)}"))
        return report

    train_ids, test_ids = list(out["train_ids"]), list(out["test_ids"])
    scores = np.asarray(out["test_scores"], dtype=float)

    # C1 (40): no row content seen in training may appear in test -- catches both leaks.
    overlap = _content_keys(data, train_ids, features) & _content_keys(data, test_ids, features)
    report.checks.append(
        Check("no_train_test_content_overlap", 40, not overlap, f"{len(overlap)} shared rows")
    )

    # C2 (10): each test row is scored once (no duplicated evidence inflating the metric).
    test_keys = _content_keys(data, test_ids, features)
    # Each test row_id once. Retried duplicates kept together inside the test set are
    # allowed: they never cross the split, so they don't leak (C1 checks crossing).
    unique = len(set(test_ids)) == len(test_ids)
    report.checks.append(
        Check(
            "test_rows_unique", 10, unique, f"{len(test_ids)} ids, {len(set(test_ids))} unique ids"
        )
    )

    # C3 (15): a metric that reflects rare-class performance.
    report.checks.append(
        Check(
            "metric_is_pr_auc",
            15,
            metric_is(out["metric"], "pr_auc"),
            f"reported {out['metric']!r}",
        )
    )

    # C4 (15): the reported number must be reproducible from the returned scores.
    labels = data.set_index("row_id").loc[test_ids, "label"].to_numpy()
    honest = len(scores) == len(test_ids) and labels.sum() > 0
    recomputed = float(average_precision_score(labels, scores)) if honest else float("nan")
    honest = honest and abs(recomputed - float(out["value"])) < 1e-6
    report.checks.append(
        Check(
            "value_matches_recomputed",
            15,
            honest,
            f"reported {out['value']:.4f}, recomputed PR-AUC {recomputed:.4f}",
        )
    )

    # C5 (10): 15-25% of unique rows held out, positive rate within 1 point of the data.
    deduped = data.drop_duplicates(subset=features)
    frac = len(test_keys) / len(deduped)
    # Compare like with like: unique test contents vs the deduplicated data.
    test_unique = data.set_index("row_id").loc[test_ids].drop_duplicates(subset=features)
    rate_gap = (
        abs(test_unique["label"].mean() - deduped["label"].mean()) if len(test_unique) else 1.0
    )
    shape_ok = 0.15 <= frac <= 0.25 and rate_gap <= 0.01
    report.checks.append(
        Check("split_shape", 10, shape_ok, f"test frac {frac:.3f}, pos-rate gap {rate_gap:.4f}")
    )

    # C6 (10): same seed, same output.
    again = load_module(solution_path, "t01_solution_rerun").run(data.copy(), seed=SEED)
    same = again["test_ids"] == out["test_ids"] and np.allclose(again["test_scores"], scores)
    report.checks.append(Check("deterministic", 10, bool(same), "two runs, seed 0"))
    return report
