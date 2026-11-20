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

**A grader is only trusted if it passes the reference, fails the starter, fails tempting partial fixes, and still accepts other honest methods.** CI enforces this on every push.

## Coverage of the leakage taxonomy

Tasks map to the eight leakage types in Kapoor & Narayanan's taxonomy, so coverage can be checked against published research rather than a hand-picked list.

| Type | Leakage | Task | Status |
|---|---|---|---|
| L1.1 | No test set (tuning or reporting on the same data) | [t05_test_set_selection](tasks/t05_test_set_selection) | done |
| L1.2 | Pre-processing on training and test set | [t01_oversample_leak](tasks/t01_oversample_leak) | done |
| L1.3 | Feature selection on training and test set | [t04_feature_selection_leak](tasks/t04_feature_selection_leak) | done |
| L1.4 | Duplicates in datasets | [t01_oversample_leak](tasks/t01_oversample_leak) | done |
| L2 | Model uses features that are not legitimate | [t02_timeseries_cv_leak](tasks/t02_timeseries_cv_leak) | done |
| L3.1 | Temporal leakage | [t02_timeseries_cv_leak](tasks/t02_timeseries_cv_leak) | done |
| L3.2 | Non-independence between train and test samples | [t03_patient_leak](tasks/t03_patient_leak) | done |
| L3.3 | Sampling bias in test distribution | [t06_regional_shift](tasks/t06_regional_shift) | done |

**All eight leakage types are covered.** Beyond leakage, later tasks will cover unstable training (NaN loss), irreproducibility (unseeded data loaders), and performance bottlenecks.

## Results

### Grader validation

| Task | Reference | Buggy starter | Partial fix |
|---|---|---|---|
| t01_oversample_leak | 100/100 | 10/100 (reports 0.998 accuracy; leaked PR-AUC 1.000) | caught (fails C1) |
| t02_timeseries_cv_leak | 100/100 (MAE 5.29) | 40/100 (MAE 4.01, at the noise floor) | caught: centred window and same-day rolling mean both fail C2 |
| t03_patient_leak | 100/100 (ROC-AUC 0.77) | 45/100 (ROC-AUC 0.93; 269 patients in both sets) | caught: last-visit holdout (ROC-AUC 0.96) fails C1 |
| t04_feature_selection_leak | 100/100 (ROC-AUC 0.73) | 60/100 (ROC-AUC 0.94; **0.92 on random labels**) | caught: constant-score "fix" fails the real-signal floor |
| t05_test_set_selection | 100/100 (ROC-AUC 0.71) | 60/100 (ROC-AUC 0.74; **0.68 on random labels**) | fair: a second honest method also scores 100/100 |
| t06_regional_shift | 100/100 (ROC-AUC 0.78) | 50/100 (ROC-AUC 0.97 on Nyeri; 0.69 in new counties) | caught: pooled CV (0.84) and best-2-counties cherry-pick (0.85) |

### AI assistant evaluation

_In progress:_ do AI coding assistants catch these leaks on their own, or only when asked to review? Each assistant gets every task four times: **realistic** (a data preview and the model code, "improve this model"), **blind** (the data generator's source and the model code), **neutral** (the scenario with its claimed score, "improve this model") and **review** ("find what makes this untrustworthy"). See the [protocol](evals/PROTOCOL.md).

A pilot run on t01 found that the original grader scored two correct fixes 65/100: it insisted on the metric name `pr_auc`, rejected retried duplicates kept inside the test set, and compared positive rates on mismatched bases. All graders now accept equivalent metric names, and both pilot answers are regression tests. Grader fairness was checked before any results were collected. Session A then surfaced one more: on t03 ChatGPT reported ROC-AUC per patient, a better practice than the reference's per-visit AUC, and the grader now accepts both.

## Run it

```bash
make setup
make validate                                              # every grader vs its reference and starter
uv run python -m bugbench.runner grade t01 my_solution.py  # grade your own attempt (t01, t02, ...)
make check                                                 # lint, types, tests
uv run python -m bugbench.prompt t01 neutral                # the exact prompt given to an assistant
make evaluate                                              # grade saved assistant answers
```

## Limitations

- Datasets are synthetic, chosen so that bugs are controllable and graders deterministic. Real-world data is messier.
- Automatic checks verify outcomes (no overlap, recomputable metric), not reasoning; written explanations are graded separately.
- Reference solutions are public, so these tasks suit practice and assistant evaluation, not a held-out leaderboard.
- Most graders trust the train/test ids a solution reports, so a solver that trains on test data and misreports its ids would pass. Behavioural checks (t02's future-scramble test, t04's random-label test) catch some cases; agent transcripts are reviewed in the agentic evaluation.

## Reference

Kapoor, S. & Narayanan, A. (2023). Leakage and the reproducibility crisis in machine-learning-based science. *Patterns*, 4(9), 100804. https://doi.org/10.1016/j.patter.2023.100804
