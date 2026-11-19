"""The grader is itself code under test: it must pass the fix and fail every flawed variant."""

from pathlib import Path

import pytest

from bugbench.core import TASKS_DIR, Report
from bugbench.runner import grade

TASK = TASKS_DIR / "t01_oversample_leak"

# A tempting partial fix: split before oversampling, but forget the natural duplicates.
PARTIAL_FIX = """
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

FEATURES = [f"f{i}" for i in range(12)]

def run(data, seed=0):
    train, test = train_test_split(data, test_size=0.2, stratify=data["label"], random_state=seed)
    pos, neg = train[train.label == 1], train[train.label == 0]
    train = pd.concat([neg, resample(pos, n_samples=len(neg), random_state=seed)])
    model = RandomForestClassifier(n_estimators=50, random_state=seed)
    model.fit(train[FEATURES], train.label)
    scores = model.predict_proba(test[FEATURES])[:, 1]
    return {"train_ids": train.row_id.tolist(), "test_ids": test.row_id.tolist(),
            "test_scores": scores.tolist(), "metric": "pr_auc",
            "value": float(average_precision_score(test.label, scores)), "notes": ""}
"""


def _failed(report: Report) -> set[str]:
    return {c.name for c in report.checks if not c.passed}


def test_reference_scores_full_marks() -> None:
    report = grade(TASK, TASK / "reference" / "solution.py")
    assert report.passed and report.score == report.max_score == 100


def test_starter_bug_is_caught() -> None:
    report = grade(TASK, TASK / "starter" / "solution.py")
    assert {"no_train_test_content_overlap", "metric_is_pr_auc"} <= _failed(report)


def test_partial_fix_still_leaks_through_duplicates(tmp_path: Path) -> None:
    path = tmp_path / "partial.py"
    path.write_text(PARTIAL_FIX)
    report = grade(TASK, path)
    assert "no_train_test_content_overlap" in _failed(report)
    assert "metric_is_pr_auc" not in _failed(report)


def test_missing_keys_fail_cleanly(tmp_path: Path) -> None:
    path = tmp_path / "empty.py"
    path.write_text("def run(data, seed=0):\n    return {}\n")
    report = grade(TASK, path)
    assert not report.passed and report.score == 0


@pytest.mark.parametrize("seed", [0, 1])
def test_data_is_deterministic(seed: int) -> None:
    from bugbench.core import load_module

    data = load_module(TASK / "data.py", "t01_data_test")
    assert data.load_data(seed).equals(data.load_data(seed))


@pytest.mark.parametrize("name", ["t01_chatgpt_review.py", "t01_chatgpt_neutral.py"])
def test_pilot_answers_that_fix_the_leak_score_full_marks(name: str) -> None:
    """Regression: the first pilot showed the grader rejecting two correct fixes
    (metric named 'average_precision'; retried duplicates kept together in the test set)."""
    report = grade(TASK, Path(__file__).parent / "fixtures" / name)
    assert report.passed and report.score == 100
