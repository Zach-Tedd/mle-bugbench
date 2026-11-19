"""Task 05 checks test-set selection leakage without rewarding one honest method."""

from pathlib import Path

from bugbench.core import TASKS_DIR, Report
from bugbench.runner import grade

TASK = TASKS_DIR / "t05_test_set_selection"

VALIDATION_SPLIT = """
import importlib.util
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

_spec = importlib.util.spec_from_file_location("t05_data", "__DATA__")
_data = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_data)
CANDIDATES = _data.candidate_subsets()

def run(data, seed=0):
    train, test = train_test_split(data, test_size=0.3, stratify=data["label"], random_state=seed)
    fit, val = train_test_split(train, test_size=0.3, stratify=train["label"], random_state=seed)
    best_auc, best_cols = -1.0, CANDIDATES[0]
    for cols in CANDIDATES:
        m = LogisticRegression(max_iter=500).fit(fit[cols], fit["label"])
        a = roc_auc_score(val["label"], m.predict_proba(val[cols])[:, 1])
        if a > best_auc:
            best_auc, best_cols = a, cols
    m = LogisticRegression(max_iter=500).fit(train[best_cols], train["label"])
    s = m.predict_proba(test[best_cols])[:, 1]
    return {"train_ids": train["row_id"].tolist(), "test_ids": test["row_id"].tolist(),
            "test_scores": s.tolist(), "chosen": best_cols, "metric": "roc_auc",
            "value": float(roc_auc_score(test["label"], s)), "notes": ""}
"""

CONSTANT_SCORE = """
from sklearn.model_selection import train_test_split

def run(data, seed=0):
    train, test = train_test_split(
        data, test_size=0.3, stratify=data["label"], random_state=seed
    )
    scores = [0.5] * len(test)
    return {
        "train_ids": train["row_id"].tolist(),
        "test_ids": test["row_id"].tolist(),
        "test_scores": scores,
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


def test_separate_validation_selection_method_also_scores_full_marks(
    tmp_path: Path,
) -> None:
    path = tmp_path / "validation_split.py"
    solution = VALIDATION_SPLIT.replace("__DATA__", str(TASK / "data.py"))
    path.write_text(solution)
    report = grade(TASK, path)
    assert report.passed and report.score == report.max_score == 100


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
