"""The evaluation tooling: prompts are built faithfully and answers are graded safely."""

import shutil
from pathlib import Path

import pytest

from bugbench.core import TASKS_DIR, task_dirs
from bugbench.evaluate import evaluate
from bugbench.prompt import build_prompt


@pytest.mark.parametrize("task_dir", task_dirs(), ids=lambda d: d.name)
def test_prompts_carry_code_and_differ_only_in_the_ask(task_dir: Path) -> None:
    neutral = build_prompt(task_dir, "neutral")
    review = build_prompt(task_dir, "review")
    for prompt in (neutral, review):
        assert "def load_data" in prompt and "def run" in prompt
        assert "must define `run(" in prompt
    assert "Review it" not in neutral
    assert "wrong" not in neutral.lower().split("`data.py`:")[0], (
        "neutral prompt must not hint at a bug"
    )
    assert "Review it" in review


@pytest.mark.parametrize("task_dir", task_dirs(), ids=lambda d: d.name)
def test_blind_prompt_has_code_but_no_scenario_or_claim(task_dir: Path) -> None:
    blind = build_prompt(task_dir, "blind")
    assert "def load_data" in blind and "def run" in blind
    context = build_prompt(task_dir, "neutral").split("\n\n")[0]
    assert context not in blind, "blind prompt must not include the scenario"
    for cue in ("teammate", "reports", "wants to", "publish", "ship", "wrong", "bug", "leak"):
        assert cue not in blind.lower(), f"blind prompt contains the cue {cue!r}"


def test_unknown_condition_rejected() -> None:
    with pytest.raises(ValueError):
        build_prompt(TASKS_DIR / "t01_oversample_leak", "hint")


def test_evaluate_grades_pass_fail_and_crash(tmp_path: Path) -> None:
    task = TASKS_DIR / "t01_oversample_leak"
    demo = tmp_path / "runs" / "demo"
    demo.mkdir(parents=True)
    shutil.copyfile(task / "reference" / "solution.py", demo / "t01_review.py")
    shutil.copyfile(task / "starter" / "solution.py", demo / "t01_neutral.py")
    (demo / "t02_review.py").write_text("raise RuntimeError('model wrote broken code')\n")

    results = {(r.task[:3], r.condition): r for r in evaluate(tmp_path / "runs", tmp_path)}

    assert results[("t01", "review")].passed and results[("t01", "review")].score == 100
    assert not results[("t01", "neutral")].passed
    assert "no_train_test_content_overlap" in results[("t01", "neutral")].failed_checks
    assert results[("t02", "review")].error.startswith("crashed")
    assert (tmp_path / "results.md").exists() and (tmp_path / "results.csv").exists()
    assert not (task / "_attempt").exists()
