"""Validate graders and grade submissions.

python -m bugbench.runner validate            # every reference passes, every starter fails
python -m bugbench.runner grade t01 path.py   # grade one submission
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from bugbench.core import TASKS_DIR, Report, load_module, task_dirs


def grade(task_dir: Path, solution_path: Path) -> Report:
    grader = load_module(task_dir / "grader.py", f"{task_dir.name}_grader")
    report: Report = grader.grade(solution_path)
    return report


def validate() -> int:
    """A grader is only trustworthy if it passes the fix AND fails the bug."""
    failures = 0
    results = []
    for task_dir in task_dirs():
        ref = grade(task_dir, task_dir / "reference" / "solution.py")
        bug = grade(task_dir, task_dir / "starter" / "solution.py")
        print(ref.render())
        print(bug.render())
        if not ref.passed:
            print(f"!! {task_dir.name}: reference does not pass its own grader")
            failures += 1
        if bug.passed:
            print(f"!! {task_dir.name}: grader does not catch the planted bug")
            failures += 1
        results += [ref.as_dict(), bug.as_dict()]
    out = Path("results/validation.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bugbench")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("validate")
    g = sub.add_parser("grade")
    g.add_argument("task", help="task folder prefix, e.g. t01")
    g.add_argument("solution", type=Path)
    args = parser.parse_args(argv)

    if args.cmd == "validate":
        return 1 if validate() else 0
    matches = [d for d in task_dirs() if d.name.startswith(args.task)]
    if len(matches) != 1:
        print(f"no unique task matching {args.task!r} in {TASKS_DIR}")
        return 2
    report = grade(matches[0], args.solution)
    print(report.render())
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
