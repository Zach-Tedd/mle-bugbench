# t01 rubric

A solution **passes** only if every automatic criterion passes. Weights show relative importance.

| # | Criterion | Weight | How it's checked | What failing it means |
|---|---|---|---|---|
| C1 | No row content appears in both train and test | 40 | Hash of feature values for every train and test row; intersection must be empty | Leakage: oversampling before the split **or** natural duplicates straddling the split |
| C2 | Each test row is scored once | 10 | Test ids and test-row contents are unique | Duplicated evidence inflates the metric |
| C3 | Metric reflects rare-class performance | 15 | `metric == "pr_auc"` | Accuracy is ~97% for a model that never predicts fraud |
| C4 | Reported value is reproducible | 15 | Grader recomputes PR-AUC from `test_scores` and true labels; must match to 1e-6 | The number can't be trusted |
| C5 | Sensible held-out split | 10 | 15–25% of unique rows in test; positive rate within 1 point of the deduplicated data | Unrepresentative or unstratified test set |
| C6 | Deterministic | 10 | Two runs with seed 0 give identical ids and scores | Results can't be reproduced |

## Manual criterion (graded in Step 4 by an LLM judge, then spot-checked by a human)

| # | Criterion | Full credit |
|---|---|---|
| M1 | `notes` identify the cause | Names **both** leaks (duplicates and oversample-before-split) and explains why accuracy is misleading at a 3% base rate |

## Reference results (seed 0)

| Solution | Score | Reported metric |
|---|---|---|
| Reference | 100/100 | PR-AUC 0.655 |
| Starter (buggy) | 10/100 | accuracy 0.998 (its PR-AUC recomputes to a leaked 1.000) |
| Partial fix (split first, no dedupe) | fails C1 | see `tests/test_t01.py` |
