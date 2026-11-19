# t04 rubric

A solution **passes** only if every automatic criterion passes. Weights show relative importance.

| # | Criterion | Weight | How it's checked | What failing it means |
|---|---|---|---|---|
| C1 | **Random-label test** | 40 | Rerun the solution three times with labels shuffled (seeds 11, 22, 33), compute ROC-AUC from the returned scores, and average. Must be ≤ 0.60 | The pipeline "learns" labels that are pure noise, so some step saw validation labels |
| C2 | Folds are clean | 10 | Within each fold, train and validation ids don't overlap; every sample is validated exactly once | Invalid cross-validation |
| C3 | Reported value is reproducible | 5 | Recompute mean per-fold ROC-AUC from `val_scores`; must match to 1e-6 | The number can't be trusted |
| C4 | Enough folds | 5 | At least 3 | Unstable estimate |
| C5 | Metric is ROC-AUC | 10 | `metric` names ROC-AUC (`roc_auc`, `auc`, `auroc`) | Not comparable with the claim |
| C6 | Deterministic | 15 | Two runs on real labels give identical folds and scores | Results can't be reproduced |
| C7 | Learns real signal | 15 | Real-label ROC-AUC ≥ 0.60 | The "fix" removed the model's ability to learn (e.g. constant scores) |

**Why C1 averages three shuffles:** with only 200 samples, one shuffle can push an honest pipeline to 0.62 by chance. Across 30 shuffles the honest mean never exceeded 0.52 when averaged in threes, while the leaky pipeline never dropped below 0.89. The threshold of 0.60 leaves a wide margin on both sides.

**Why C1 tests behaviour:** it doesn't trust the solution's reported ids or numbers. It checks what the pipeline does when there is nothing to learn, so it catches selection, scaling or any other step fitted on validation labels, however the code is written.

## Manual criterion (graded in Step 4 by an LLM judge, then spot-checked by a human)

| # | Criterion | Full credit |
|---|---|---|
| M1 | `notes` identify the cause | Explains that selecting genes on all samples lets validation labels influence the features, and that the fix is to refit selection inside each training fold |

## Reference results (seed 0)

| Solution | Score | Real-label ROC-AUC | Random-label ROC-AUC |
|---|---|---|---|
| Reference (selection inside a per-fold pipeline) | 100/100 | 0.730 | 0.513 |
| Starter (selection on all samples, then CV) | 60/100, fails C1 | 0.945 | **0.919** |
| Constant scores (model removed) | 85/100, fails C7 | 0.500 | 0.500 |

Honest alternatives also pass C7: no selection (0.72), k = 10 (0.85), k = 100 (0.66).

## Grader history

The first version had no C7, so a solution returning 0.5 for every sample scored 100/100: it is trivially immune to the random-label test. C1 also first read the solution's reported value; it now recomputes AUC from the returned scores. `tests/test_t04.py` keeps the constant-score case as a regression test.
