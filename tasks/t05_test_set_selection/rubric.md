# t05 rubric

A solution **passes** only if every automatic criterion passes. Weights show relative importance.

| # | Criterion | Weight | How it's checked | What failing it means |
|---|---|---|---|---|
| C1 | **Random-label test** | 40 | Rerun the solution three times with labels shuffled (seeds 11, 22, 33), compute ROC-AUC from the returned test scores against the shuffled labels, and average. Must be ≤ 0.60 | The reported score was selected on the test set, so it reflects noise that happened to fit |
| C2 | Clean split | 10 | Train and test ids don't overlap; test ids are unique; the test set holds 25–35% of samples | No genuine held-out set |
| C3 | Reported value is reproducible | 10 | Recompute ROC-AUC from `test_scores`; must match to 1e-6 | The number can't be trusted |
| C4 | Metric is ROC-AUC | 10 | `metric == "roc_auc"` | Not comparable with the claim |
| C5 | Deterministic | 15 | Two runs with seed 0 give identical ids, scores and value | Results can't be reproduced |
| C6 | Learns real signal | 15 | Real-label ROC-AUC ≥ 0.60 | The "fix" removed the model's ability to learn |

**Why this leak is hard to see on real labels:** choosing by test score inflates the result only modestly here (0.74 vs 0.71). The random-label test exposes it: with 200 candidates, choosing by test score still "finds" a model scoring about 0.68 on labels that are pure noise. With grader seeds 11, 22, 33 the starter averages 0.679 and the reference 0.508, either side of the 0.60 threshold.

## Manual criterion (graded in Step 4 by an LLM judge, then spot-checked by a human)

| # | Criterion | Full credit |
|---|---|---|
| M1 | `notes` identify the cause | Explains that selecting among many models by test score turns the test set into a selection set, and that the fix is to select on training data only and evaluate the test set once |

## Reference results (seed 0)

| Solution | Score | Real-label ROC-AUC | Random-label ROC-AUC |
|---|---|---|---|
| Reference (select by 5-fold CV on training data, test once) | 100/100 | 0.713 | 0.508 |
| Alternative honest method (select on a separate validation split, test once) | 100/100 | 0.727 | 0.517 |
| Starter (select by test score) | 60/100, fails C1 | 0.744 | **0.679** |
| Constant scores (model removed) | 85/100, fails C6 | 0.500 | 0.500 |

## Fairness

A grader that rewards only the reference's exact method would penalise correct alternatives. `tests/test_t05.py` checks that a second honest method (validation-split selection) also scores 100/100.
