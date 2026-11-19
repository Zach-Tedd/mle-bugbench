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
REQUIRED_KEYS = {"folds", "metric", "value", "notes"}


def grade(solution_path: Path) -> Report:
    data_mod = load_module(TASK_DIR / "data.py", "t04_data")
    data: pd.DataFrame = data_mod.load_data(SEED)
    report = Report(task=TASK_DIR.name, solution=str(solution_path))

    solution = load_module(solution_path, "t04_solution")
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
                f"t04_solution_shuffle_{shuffle_seed}",
            ).run(shuffled, seed=SEED)
            shuffled_labels = shuffled.set_index("row_id")["label"]
            fold_aucs = []
            for fold in shuffled_out["folds"]:
                val_ids = list(fold["val_ids"])
                scores = np.asarray(fold["val_scores"], dtype=float)
                labels = shuffled_labels.loc[val_ids].to_numpy()
                if len(scores) != len(val_ids) or not np.isfinite(scores).all():
                    raise ValueError("prediction count mismatch or non-finite scores")
                if len(np.unique(labels)) != 2:
                    raise ValueError("validation fold must contain both label classes")
                fold_aucs.append(roc_auc_score(labels, scores))
            if not fold_aucs:
                raise ValueError("solution returned no validation folds")
            shuffled_aucs.append(float(np.mean(fold_aucs)))
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

    folds = out["folds"] if isinstance(out["folds"], list) else []
    expected_ids = set(data["row_id"].tolist())
    validation_ids: list[int] = []
    folds_clean = len(folds) > 0
    try:
        for fold in folds:
            train_ids = list(fold["train_ids"])
            val_ids = list(fold["val_ids"])
            if set(train_ids) & set(val_ids):
                folds_clean = False
            validation_ids.extend(val_ids)
        folds_clean = folds_clean and set(validation_ids) == expected_ids
        folds_clean = folds_clean and len(validation_ids) == len(expected_ids)
    except (KeyError, TypeError, ValueError):
        folds_clean = False
    report.checks.append(
        Check(
            "folds_are_clean",
            10,
            folds_clean,
            "train/validation IDs disjoint within folds; each sample validated once",
        )
    )

    reproducible = False
    recomputed = float("nan")
    try:
        labels_by_id = data.set_index("row_id")["label"]
        fold_aucs = []
        for fold in folds:
            val_ids = list(fold["val_ids"])
            scores = np.asarray(fold["val_scores"], dtype=float)
            labels = labels_by_id.loc[val_ids].to_numpy()
            if len(scores) != len(val_ids) or not np.isfinite(scores).all():
                raise ValueError("prediction count mismatch or non-finite scores")
            if len(np.unique(labels)) != 2:
                raise ValueError("validation fold must contain both label classes")
            fold_aucs.append(roc_auc_score(labels, scores))
        recomputed = float(np.mean(fold_aucs))
        reproducible = bool(
            np.isfinite(recomputed) and abs(recomputed - float(out["value"])) < 1e-6
        )
    except (KeyError, TypeError, ValueError, IndexError):
        reproducible = False
    report.checks.append(
        Check(
            "value_matches_recomputed",
            5,
            reproducible,
            f"reported {float(out['value']):.4f}, recomputed mean ROC-AUC {recomputed:.6f}",
        )
    )

    enough_folds = len(folds) >= 3
    report.checks.append(
        Check("at_least_three_folds", 5, enough_folds, f"reported {len(folds)} folds")
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
        again = load_module(solution_path, "t04_solution_rerun").run(data.copy(), seed=SEED)
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
    report.checks.append(
        Check("deterministic", 15, bool(deterministic), "two runs on real labels, seed 0")
    )

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
            learns_signal,
            f"real-label AUC {real_auc:.3f} (must be >= 0.60)",
        )
    )
    return report
