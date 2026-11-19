# t02 rubric

A solution **passes** only if every automatic criterion passes. Weights show relative importance.

| # | Criterion | Weight | How it's checked | What failing it means |
|---|---|---|---|---|
| C1 | Every fold trains only on the past | 30 | In each fold, the latest training date must be earlier than the earliest validation date | The model learned from days it is later tested on (shuffled K-fold) |
| C2 | No feature uses same-day or future demand | 30 | Pick a probe day in the middle of the earliest validation window. Replace demand on and after that day with random values and rerun the solution. Predictions for every validation day up to and including the probe day must not change | A feature contains the answer: a centred window, or a rolling mean that includes today |
| C3 | Reported MAE is reproducible | 15 | Recompute the mean per-fold MAE from `val_preds` and true demand; must match to 1e-6 | The number can't be trusted |
| C4 | Enough folds | 5 | At least 3 | Too few folds for a stable estimate |
| C5 | Right metric | 10 | `metric == "mae"` | Not comparable with the task's target |
| C6 | Deterministic | 10 | Two runs with seed 0 give identical folds and predictions | Results can't be reproduced |

C2 checks behaviour rather than code, so it catches look-ahead features however they're written.

## Manual criterion (graded in Step 4 by an LLM judge, then spot-checked by a human)

| # | Criterion | Full credit |
|---|---|---|
| M1 | `notes` identify the cause | Names **both** leaks (shuffled folds; a feature that includes current or future demand) and explains why an MAE at the noise floor is a warning sign |

## Reference results (seed 0)

| Solution | Score | Mean MAE |
|---|---|---|
| Reference (time-ordered folds, past-only rolling mean) | 100/100 | 5.29 |
| Starter (shuffled K-fold, centred rolling mean) | 40/100 | 4.01 |
| Partial fix (time-ordered folds, centred rolling mean) | 70/100, fails C2 | 4.23 |
| Same-day leak (time-ordered folds, rolling mean including today) | 70/100, fails C2 | 4.25 |

The irreducible-noise floor for this data is about 3.99 MAE (noise sd 5). The starter's 4.01 sits on that floor, which is itself a sign of leakage.

## Grader history

The first version of C2 scrambled demand only after the whole validation window, so the same-day leak scored 100/100. C2 now scrambles from a probe day inside the window, and `tests/test_t02.py` keeps that case as a regression test.
