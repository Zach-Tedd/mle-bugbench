# mle-bugbench

![ci](https://github.com/Zach-Tedd/mle-bugbench/actions/workflows/ci.yml/badge.svg)

**Data leakage makes machine-learning models look better than they are.** A Princeton survey found leakage in at least 294 published studies across 17 scientific fields; once the leaks were fixed, many models reported as breakthroughs did no better than decades-old logistic regression ([Kapoor & Narayanan, *Patterns*, 2023](https://doi.org/10.1016/j.patter.2023.100804)).

AI coding assistants now write a growing share of ML code, and engineers are asked to review it. **mle-bugbench measures whether those assistants, and the people reviewing them, catch the bugs that silently invalidate ML results.**

## Who it's for

- **Teams adopting AI coding assistants for ML work:** check whether the assistant notices a leaky evaluation before it reaches a report or production.
- **AI-training and evaluation teams:** ready-made tasks with objective, automatic graders.
- **Engineers and students:** practise spotting the failures that textbooks mention and real code still contains.

## What's in each task

Every task is a realistic scenario where an ML result is wrong for a subtle reason. It ships with:

| File | Purpose |
|---|---|
| `README.md` | Instructions shown to the solver, and the required interface |
| `starter/solution.py` | Plausible code containing the planted bug |
| `reference/solution.py` | The correct fix |
| `rubric.md` | Weighted, objectively checkable criteria |
| `grader.py` | Runs a solution and returns a scored report |
| `data.py` | Deterministic data, no downloads |

**A grader is only trusted if it passes the reference, fails the starter, and also fails tempting partial fixes.** CI enforces all three on every push.

## Coverage of the leakage taxonomy

Tasks map to the eight leakage types in Kapoor & Narayanan's taxonomy, so coverage can be checked against published research rather than a hand-picked list.

| Type | Leakage | Task | Status |
|---|---|---|---|
| L1.1 | No test set (tuning or reporting on the same data) | planned | |
| L1.2 | Pre-processing on training and test set | [t01_oversample_leak](tasks/t01_oversample_leak) | done |
| L1.3 | Feature selection on training and test set | planned | |
| L1.4 | Duplicates in datasets | [t01_oversample_leak](tasks/t01_oversample_leak) | done |
| L2 | Model uses features that are not legitimate | [t02_timeseries_cv_leak](tasks/t02_timeseries_cv_leak) | done |
| L3.1 | Temporal leakage | [t02_timeseries_cv_leak](tasks/t02_timeseries_cv_leak) | done |
| L3.2 | Non-independence between train and test samples | [t03_patient_leak](tasks/t03_patient_leak) | done |
| L3.3 | Sampling bias in test distribution | planned | |

Beyond leakage, later tasks cover unstable training (NaN loss), irreproducibility (unseeded data loaders), and performance bottlenecks.

## Results

### Grader validation

| Task | Reference | Buggy starter | Partial fix |
|---|---|---|---|
| t01_oversample_leak | 100/100 | 10/100 (reports 0.998 accuracy; leaked PR-AUC 1.000) | caught (fails C1) |
| t02_timeseries_cv_leak | 100/100 (MAE 5.29) | 40/100 (MAE 4.01, at the noise floor) | caught: centred window and same-day rolling mean both fail C2 |
| t03_patient_leak | 100/100 (ROC-AUC 0.77) | 45/100 (ROC-AUC 0.93; 269 patients in both sets) | caught: last-visit holdout (ROC-AUC 0.96) fails C1 |

### AI assistant evaluation

_Coming once eight tasks are in place:_ pass rates of current AI coding assistants on each task, with a failure analysis of what they missed and why.

## Run it

```bash
make setup
make validate                                              # every grader vs its reference and starter
uv run python -m bugbench.runner grade t01 my_solution.py  # grade your own attempt (t01, t02, ...)
make check                                                 # lint, types, tests
```

## Limitations

- Datasets are synthetic, chosen so that bugs are controllable and graders deterministic. Real-world data is messier.
- Automatic checks verify outcomes (no overlap, recomputable metric), not reasoning; written explanations are graded separately.
- Reference solutions are public, so these tasks suit practice and assistant evaluation, not a held-out leaderboard.
- Most graders trust the train/test ids a solution reports, so a solver that trains on test data and misreports its ids would pass. Behavioural checks (t02's future-scramble test) catch some cases; agent transcripts are reviewed in the agentic evaluation.

## Reference

Kapoor, S. & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*, 4(9), 100804. https://doi.org/10.1016/j.patter.2023.100804
