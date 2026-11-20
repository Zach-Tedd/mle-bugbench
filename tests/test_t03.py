"""The Task 03 grader must detect leakage across repeated patient visits."""

from pathlib import Path

import pytest

from bugbench.core import TASKS_DIR, Report
from bugbench.runner import grade

TASK = TASKS_DIR / "t03_patient_leak"

LAST_VISIT_PARTIAL = """
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

FEATURES = [f"f{i}" for i in range(10)]

def run(data, seed=0):
    test = data.groupby("patient_id").tail(1)
    train = data.drop(test.index)
    model = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
    model.fit(train[FEATURES], train["label"])
    scores = model.predict_proba(test[FEATURES])[:, 1]
    return {
        "train_ids": train["row_id"].tolist(),
        "test_ids": test["row_id"].tolist(),
        "test_scores": scores.tolist(),
        "metric": "roc_auc",
        "value": float(roc_auc_score(test["label"], scores)),
        "notes": "",
    }
"""


def _failed(report: Report) -> set[str]:
    return {check.name for check in report.checks if not check.passed}


def test_reference_scores_full_marks() -> None:
    report = grade(TASK, TASK / "reference" / "solution.py")
    assert report.passed and report.score == report.max_score == 100


def test_starter_fails_exact_patient_leakage_checks() -> None:
    report = grade(TASK, TASK / "starter" / "solution.py")
    assert _failed(report) == {"no_patient_overlap", "sensible_patient_split"}


def test_holding_out_last_visit_per_patient_still_leaks(
    tmp_path: Path,
) -> None:
    path = tmp_path / "last_visit_partial.py"
    path.write_text(LAST_VISIT_PARTIAL)
    report = grade(TASK, path)
    assert _failed(report) == {"no_patient_overlap", "sensible_patient_split"}


def test_empty_solution_scores_zero(tmp_path: Path) -> None:
    path = tmp_path / "empty.py"
    path.write_text("def run(data, seed=0):\n    return {}\n")
    report = grade(TASK, path)
    assert not report.passed and report.score == 0


@pytest.mark.parametrize("name", ["t03_chatgpt_neutral.py", "t03_chatgpt_review.py"])
def test_patient_level_auc_answers_score_full_marks(name: str) -> None:
    """Regression: ChatGPT reported 'patient_level_roc_auc' (one score per patient),
    which the first grader rejected because it only recomputed AUC per visit."""
    report = grade(TASK, Path(__file__).parent / "fixtures" / name)
    assert report.passed and report.score == 100
