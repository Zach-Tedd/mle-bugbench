# t03 rubric

A solution **passes** only if every automatic criterion passes. Weights show relative importance.

| # | Criterion | Weight | How it's checked | What failing it means |
|---|---|---|---|---|
| C1 | No patient appears in both train and test | 40 | Patients behind `train_ids` and `test_ids` must not intersect, and both sets must be non-empty | The model can recognise a patient's "fingerprint" instead of the disease |
| C2 | Each test row is scored once | 10 | `test_ids` are unique | Duplicated evidence inflates the metric |
| C3 | Metric is ROC-AUC | 10 | `metric == "roc_auc"` | Not comparable with the reported claim |
| C4 | Reported value is reproducible | 15 | Recompute ROC-AUC from `test_scores` and true labels; must match to 1e-6 | The number can't be trusted |
| C5 | Sensible patient-level split | 15 | 15–25% of **patients** in test; share of diseased patients in test within 5 points of the whole cohort (measured per patient, not per visit) | Unrepresentative test set |
| C6 | Deterministic | 10 | Two runs with seed 0 give identical ids, scores and value | Results can't be reproduced |

## Manual criterion (graded in Step 4 by an LLM judge, then spot-checked by a human)

| # | Criterion | Full credit |
|---|---|---|
| M1 | `notes` identify the cause | Explains that visits from one patient are not independent, so splitting by visit lets the model memorise patients; mentions that a ~80-patient test set gives a noisy AUC |

## Reference results (seed 0)

| Solution | Score | ROC-AUC | Why |
|---|---|---|---|
| Reference (`StratifiedGroupKFold` by patient) | 100/100 | 0.768 | Honest estimate for new patients |
| Starter (random split by visit) | 45/100, fails C1 and C5 | 0.935 | 269 patients in both train and test |
| Last-visit holdout (earlier visits train, last visit tests) | 45/100, fails C1 and C5 | 0.959 | Every test patient's earlier visits are in training |

The last-visit split feels like a time-based split, which makes it a common real-world mistake. It is the most inflated of the three, because **every** test patient has been seen in training.

## Known limitation

The grader checks the `train_ids` a solution reports. A solution that trained on test patients but reported clean ids would not be caught here.
