from bugbench.core import TASKS_DIR
from bugbench.runner import main


def test_grade_cli_passes_reference() -> None:
    ref = TASKS_DIR / "t01_oversample_leak" / "reference" / "solution.py"
    assert main(["grade", "t01", str(ref)]) == 0


def test_grade_cli_rejects_unknown_task() -> None:
    assert main(["grade", "t99", "x.py"]) == 2
