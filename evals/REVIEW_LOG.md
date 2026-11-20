# Failure review log

Every failed or errored run is read and classified before results are published. Graders are frozen at tag `graders-v1`. A grader or harness change is allowed only when it rejects a correct answer, and then every answer is re-graded.

**Classes**

- **Real miss**: the leak is still there.
- **Code failure**: the leak was handled, but the code crashes, so the run scores 0.
- **Context-dependent**: the prompt doesn't say where the model will be deployed, so the expected evaluation isn't defined by the prompt alone (t06 under `realistic` and `blind`).
- **Harness limit**: correct answer rejected by the evaluation setup, not the grader. Fixed and re-graded.

## Realistic condition (data preview, no generator source)

| Assistant | Task | First score | Class | Evidence |
|---|---|---|---|---|
| ChatGPT | t01 | 20 | Real miss | Removed the oversampling (the bug visible in the starter) but split rows without removing duplicates: 100 duplicate transactions in both train and test. Reported ROC-AUC rather than PR-AUC, so C3 and C4 also fail. |
| Claude | t01 | 50 | Real miss | Same pattern: fixed oversampling, split rows without deduplicating, 100 shared rows. A comment in its code says the test set "has no duplicated rows", which is false. |
| Claude | t03 | error | Code failure | Patient-level split is correct, but `df.groupby("patient_id")[BASE]` followed by `g[BASE[0]]` raises `IndexError: Column(s) … already selected`. Reproduced on pandas 2.2.3 and 3.1.1, so not version-specific. |
| ChatGPT | t05 | error (timeout) | Harness limit, then **pass** | Selection uses training-set cross-validation only (correct). About 21,000 model fits per run; 678 s for the grader's 5 runs on the review machine, over 900 s on the 4-core laptop. 100/100 with no time limit. Time limit raised to 60 minutes for all answers. |
| ChatGPT | t06 | 50 | Context-dependent | Kept training and validation on Nyeri only ("preserve the original evaluation population"). Same behaviour as its `blind` answer. |
| Claude | t06 | 70 | Context-dependent (partial) | Widened to all 5 counties with one-hot region features, but pooled them in stratified K-fold, so every county is in both training and validation. |

## Earlier conditions

| Assistant | Task | Condition | Class | Evidence |
|---|---|---|---|---|
| ChatGPT | t06 | blind | Context-dependent | Described "a strong Nyeri-specific signal in f3" but kept the Nyeri-only evaluation. |
| ChatGPT | t02 | review | Protocol slip, recorded | Answer file was empty at first grading; saved afterwards from the same chat (see `runs/chatgpt/meta.json`). |

Grader flaws found and fixed during the evaluation are listed in [PROTOCOL.md](PROTOCOL.md).
