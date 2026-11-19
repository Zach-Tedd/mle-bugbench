"""Task 04 tests guard against global feature-selection leakage and trivial scores."""

from pathlib import Path

from bugbench.core import TASKS_DIR, Report
from bugbench.runner import grade

TASK = TASKS_DIR / "t04_feature_selection_leak"

CONSTANT_SCORE = """
import numpy as np
from sklearn.model_selection import StratifiedKFold

FEATURES = [f"g{i}" for i in range(2000)]

def run(data, seed=0):
    labels = data["label"].to_numpy()
    folds = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for train_idx, val_idx in splitter.split(data[FEATURES].to_numpy(), labels):
        scores = np.full(len(val_idx), 0.5)
        folds.append({
            "train_ids": data["row_id"].iloc[train_idx].tolist(),
            "val_ids": data["row_id"].iloc[val_idx].tolist(),
            "val_scores": scores.tolist(),
        })
    return {
        "folds": folds,
        "metric": "roc_auc",
        "value": 0.5,
        "notes": "",
    }
"""


def _failed(report: Report) -> set[str]:
    return {check.name for check in report.checks if not check.passed}


def test_reference_scores_full_marks() -> None:
    report = grade(TASK, TASK / "reference" / "solution.py")
    assert report.passed and report.score == report.max_score == 100


def test_starter_fails_exactly_random_label_test() -> None:
    report = grade(TASK, TASK / "starter" / "solution.py")
    assert _failed(report) == {"random_label_test"}


def test_constant_scores_fail_only_real_signal_check(tmp_path: Path) -> None:
    path = tmp_path / "constant_score.py"
    path.write_text(CONSTANT_SCORE)
    report = grade(TASK, path)
    assert _failed(report) == {"learns_real_signal"}


def test_empty_solution_scores_zero(tmp_path: Path) -> None:
    path = tmp_path / "empty.py"
    path.write_text("def run(data, seed=0):\n    return {}\n")
    report = grade(TASK, path)
    assert not report.passed and report.score == 0
