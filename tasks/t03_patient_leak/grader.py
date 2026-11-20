from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from bugbench.core import Check, Report, load_module, metric_is

TASK_DIR = Path(__file__).resolve().parent
SEED = 0
REQUIRED_KEYS = {"train_ids", "test_ids", "test_scores", "metric", "value", "notes"}


def grade(solution_path: Path) -> Report:
    data_mod = load_module(TASK_DIR / "data.py", "t03_data")
    data: pd.DataFrame = data_mod.load_data(SEED)
    report = Report(task=TASK_DIR.name, solution=str(solution_path))

    solution = load_module(solution_path, "t03_solution")
    out: dict[str, Any] = solution.run(data.copy(), seed=SEED)

    missing = REQUIRED_KEYS - out.keys()
    if missing:
        report.checks.append(Check("interface", 100, False, f"missing keys {sorted(missing)}"))
        return report

    train_ids = list(out["train_ids"])
    test_ids = list(out["test_ids"])
    scores = np.asarray(out["test_scores"], dtype=float)
    rows = data.set_index("row_id")

    no_patient_overlap = False
    try:
        train_patients = set(rows.loc[train_ids, "patient_id"])
        test_patients = set(rows.loc[test_ids, "patient_id"])
        overlap = train_patients & test_patients
        no_patient_overlap = bool(train_patients and test_patients and not overlap)
        overlap_detail = f"{len(overlap)} patients shared"
    except (KeyError, TypeError, ValueError):
        overlap_detail = "invalid or missing row IDs"
    report.checks.append(Check("no_patient_overlap", 40, no_patient_overlap, overlap_detail))

    unique_test_rows = len(set(test_ids)) == len(test_ids)
    report.checks.append(
        Check("test_rows_unique", 10, unique_test_rows, f"{len(test_ids)} test IDs")
    )

    report.checks.append(
        Check(
            "metric_is_roc_auc",
            10,
            metric_is(out["metric"], "roc_auc"),
            f"reported {out['metric']!r}",
        )
    )

    value_matches = False
    recomputed = float("nan")
    recomputed_patient = float("nan")
    try:
        labels = rows.loc[test_ids, "label"].to_numpy()
        valid = (
            len(scores) == len(test_ids)
            and np.isfinite(scores).all()
            and len(np.unique(labels)) == 2
        )
        if valid:
            # Accept ROC-AUC per visit or per patient (mean score of each patient's
            # visits). Per-patient is the clinically standard unit: each patient counts once.
            recomputed = float(roc_auc_score(labels, scores))
            per_patient = (
                pd.DataFrame(
                    {
                        "patient": rows.loc[test_ids, "patient_id"].to_numpy(),
                        "score": scores,
                        "label": labels,
                    }
                )
                .groupby("patient")
                .agg(score=("score", "mean"), label=("label", "first"))
            )
            recomputed_patient = float(roc_auc_score(per_patient["label"], per_patient["score"]))
            reported = float(out["value"])
            value_matches = (
                min(abs(recomputed - reported), abs(recomputed_patient - reported)) < 1e-6
            )
    except (KeyError, TypeError, ValueError):
        labels = np.asarray([])
    report.checks.append(
        Check(
            "value_matches_recomputed",
            15,
            value_matches,
            f"reported {float(out['value']):.4f}, recomputed ROC-AUC per visit "
            f"{recomputed:.6f}, per patient {recomputed_patient:.6f}",
        )
    )

    sensible_split = False
    try:
        test_patient_ids = set(rows.loc[test_ids, "patient_id"])
        patient_labels = data.groupby("patient_id")["label"].first()
        test_patients = patient_labels.loc[list(test_patient_ids)]
        test_fraction = len(test_patient_ids) / len(patient_labels)
        prevalence_gap = abs(float(test_patients.mean()) - float(patient_labels.mean()))
        sensible_split = 0.15 <= test_fraction <= 0.25 and prevalence_gap <= 0.05
        split_detail = (
            f"test patient fraction {test_fraction:.3f}, disease-rate gap {prevalence_gap:.4f}"
        )
    except (KeyError, TypeError, ValueError):
        split_detail = "could not compute patient split shape"
    report.checks.append(Check("sensible_patient_split", 15, sensible_split, split_detail))

    deterministic = False
    try:
        again = load_module(solution_path, "t03_solution_rerun").run(data.copy(), seed=SEED)
        deterministic = (
            again["train_ids"] == train_ids
            and again["test_ids"] == test_ids
            and np.allclose(again["test_scores"], scores)
            and np.isclose(float(again["value"]), float(out["value"]))
        )
    except (KeyError, TypeError, ValueError):
        deterministic = False
    report.checks.append(Check("deterministic", 10, bool(deterministic), "two runs, seed 0"))
    return report
