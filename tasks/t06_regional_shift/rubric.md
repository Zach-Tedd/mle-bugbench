# t06 rubric

A solution **passes** only if every automatic criterion passes. Weights show relative importance.

| # | Criterion | Weight | How it's checked | What failing it means |
|---|---|---|---|---|
| C1 | Validation counties are unseen | 30 | In every fold, no county appears in both the training and validation rows; failures name the shared counties | The test set comes from the same places as training, so it can't reveal shortcuts that fail elsewhere |
| C2 | Validation covers the deployment area | 20 | Across all folds, at least 4 of the 5 counties are validated | The estimate rests on a hand-picked subset of counties |
| C3 | Reported value is reproducible | 15 | Recompute mean per-fold ROC-AUC from `val_scores`; must match to 1e-6 | The number can't be trusted |
| C4 | Metric is ROC-AUC | 5 | `metric` names ROC-AUC (`roc_auc`, `auc`, `auroc`) | Not comparable with the claim |
| C5 | Deterministic | 15 | Two runs with seed 0 give identical folds, scores and value | Results can't be reproduced |
| C6 | Learns real signal | 15 | Mean ROC-AUC ≥ 0.60 | The "fix" removed the model's ability to learn |

**Why the starter looks so good:** on Nyeri farms, diseased plants were often photographed in shade, so one feature is a strong but local shortcut. A model trained and tested on Nyeri learns it (0.97); the same model scores 0.69 in the other four counties.

**Why C2 requires 4 of 5 counties, not 2 or 5:** requiring all 5 would fail a legitimate external validation (train on Nyeri, test on the other four). Requiring only 2 let a solver run leave-one-county-out honestly and then report only its two best counties. At 4 of 5, dropping even the weakest county moves the mean only from 0.780 to 0.796.

## Manual criterion (graded in Step 4 by an LLM judge, then spot-checked by a human)

| # | Criterion | Full credit |
|---|---|---|
| M1 | `notes` identify the cause | Explains that a test set drawn from the training county can't detect a county-specific shortcut, and that validation must use counties held out from training |

## Reference results (seed 0)

| Solution | Score | Mean ROC-AUC |
|---|---|---|
| Reference (leave one county out; all 5 validated) | 100/100 | 0.780 |
| Fair alternative (train on Nyeri, test on the other 4) | 100/100 | 0.689 |
| Starter (train and test on Nyeri) | 50/100, fails C1 and C2 | 0.971 |
| Pooled 5-fold CV across all counties | fails C1 only | 0.838 |
| Leave one county out, best 2 folds kept (Meru, Nyeri) | fails C2 only | 0.845 |

Per-county results from the reference: Embu 0.767, Kiambu 0.717, Kirinyaga 0.725, Meru 0.856, Nyeri 0.835.

## Grader history

C2 first required only 2 validated counties, so cherry-picking the two best folds scored 100/100. It now requires 4 of 5, and `tests/test_t06.py` keeps the cherry-pick as a regression test.
