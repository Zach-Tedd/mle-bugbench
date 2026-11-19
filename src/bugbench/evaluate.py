"""Grade every saved AI-assistant answer and write a results table.

Layout (one folder per assistant):

    evals/runs/<assistant>/meta.json          {"model": "...", "interface": "...", "date": "..."}
    evals/runs/<assistant>/t01_neutral.py     the solution.py the assistant returned
    evals/runs/<assistant>/t01_review.py

    uv run python -m bugbench.evaluate        -> evals/results.csv and evals/results.md

Model-written code runs with your permissions: read each file before grading it.
Each grade runs in a separate process with a time limit, so one hung answer
can't stall the whole evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from bugbench.core import task_dirs
from bugbench.prompt import CONDITIONS

TIMEOUT_S = 900
ATTEMPT_DIR = "_attempt"


@dataclass
class Result:
    assistant: str
    task: str
    condition: str
    score: int
    passed: bool
    failed_checks: str
    error: str


def _grade_isolated(task_dir: Path, answer: Path, timeout_s: int) -> dict[str, object]:
    """Copy the answer into the task folder (so relative imports like the starter's work),
    grade it in a subprocess, and always clean up."""
    attempt = task_dir / ATTEMPT_DIR
    attempt.mkdir(exist_ok=True)
    target = attempt / "solution.py"
    shutil.copyfile(answer, target)
    try:
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "bugbench.runner",
                "grade",
                task_dir.name,
                str(target),
                "--json",
            ],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"error": f"timed out after {timeout_s}s"}
    finally:
        shutil.rmtree(attempt, ignore_errors=True)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
    if not lines:
        tail = (proc.stderr.strip().splitlines() or ["no output"])[-1]
        return {"error": f"crashed: {tail}"[:200]}
    report: dict[str, object] = json.loads(lines[-1])
    return report


def evaluate(runs_dir: Path, out_dir: Path, timeout_s: int = TIMEOUT_S) -> list[Result]:
    tasks = {d.name[:3]: d for d in task_dirs()}
    results: list[Result] = []
    for assistant_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
        for answer in sorted(assistant_dir.glob("t??_*.py")):
            prefix, condition = answer.stem.split("_", 1)
            if prefix not in tasks or condition not in CONDITIONS:
                print(f"skipping {answer}: expected tNN_<{'|'.join(CONDITIONS)}>.py")
                continue
            print(f"grading {assistant_dir.name}/{answer.name} ...", flush=True)
            report = _grade_isolated(tasks[prefix], answer, timeout_s)
            checks = report.get("checks", [])
            failed = (
                [c["name"] for c in checks if not c["passed"]] if isinstance(checks, list) else []
            )
            results.append(
                Result(
                    assistant=assistant_dir.name,
                    task=tasks[prefix].name,
                    condition=condition,
                    score=int(report.get("score", 0)),  # type: ignore[call-overload]
                    passed=bool(report.get("passed", False)),
                    failed_checks=";".join(failed),
                    error=str(report.get("error", "")),
                )
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(results, out_dir / "results.csv")
    _write_markdown(results, out_dir / "results.md")
    return results


def _write_csv(results: list[Result], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["assistant", "task", "condition", "score", "passed", "failed_checks", "error"]
        )
        for r in results:
            writer.writerow(
                [r.assistant, r.task, r.condition, r.score, r.passed, r.failed_checks, r.error]
            )


def _write_markdown(results: list[Result], path: Path) -> None:
    assistants = sorted({r.assistant for r in results})
    tasks = sorted({r.task for r in results})
    lookup = {(r.assistant, r.task, r.condition): r for r in results}
    header = ["Task"] + [f"{a} ({c})" for a in assistants for c in CONDITIONS]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for task in tasks:
        cells = [task]
        for a in assistants:
            for c in CONDITIONS:
                r = lookup.get((a, task, c))
                if r is None:
                    cells.append("–")
                elif r.error:
                    cells.append("error")
                else:
                    cells.append(f"{'pass' if r.passed else 'fail'} ({r.score})")
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "| Assistant | Condition | Passed | Mean score |", "|---|---|---|---|"]
    for a in assistants:
        for c in CONDITIONS:
            rs = [r for r in results if r.assistant == a and r.condition == c]
            if rs:
                n_pass = sum(r.passed for r in rs)
                mean = sum(r.score for r in rs) / len(rs)
                lines.append(f"| {a} | {c} | {n_pass}/{len(rs)} | {mean:.0f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bugbench.evaluate")
    parser.add_argument("--runs", type=Path, default=Path("evals/runs"))
    parser.add_argument("--out", type=Path, default=Path("evals"))
    parser.add_argument("--timeout", type=int, default=TIMEOUT_S)
    args = parser.parse_args(argv)
    if not args.runs.is_dir():
        print(f"no runs folder at {args.runs}")
        return 2
    results = evaluate(args.runs, args.out, args.timeout)
    print((args.out / "results.md").read_text(encoding="utf-8"))
    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
