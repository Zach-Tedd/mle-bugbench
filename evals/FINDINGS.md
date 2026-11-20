# Findings: do AI coding assistants catch ML data leakage?

**Short answer: they fix leakage they can see in the code, and miss leakage that only shows in the data or depends on how the model will be used.** Given a realistic view of a project (a data preview and the model code), ChatGPT and Claude produced leak-free solutions in 7 of 12 runs. When the code that generated the data was also visible, which reveals how each bug was planted, that rose to 11 of 12.

These are case studies from one run per task and condition, not measured rates. See [Limitations](#limitations).

## Setup

- **Tasks:** the 6 tasks in this benchmark, which together cover all 8 leakage types in Kapoor & Narayanan's taxonomy.
- **Assistants:** ChatGPT (free tier, web chat; the model name was not displayed) and Claude Sonnet 5.5 (web chat). Default settings, memory off, a new chat for every run, one message, no retries.
- **Conditions**, from least to most information:

| Condition | The assistant sees | The ask |
|---|---|---|
| `realistic` | A data preview (size, column types, first rows, summaries) and the model code | "Improve this model" |
| `blind` | The data generator's source code and the model code | "Improve this model" |
| `neutral` | The scenario (including the claimed score), the generator source and the model code | "Improve this model" |
| `review` | Same as `neutral` | "Fix every problem that makes the result untrustworthy" |

- **Grading:** automatic, by the graders frozen at tag `graders-v1`, with every failure read and classified in the [review log](REVIEW_LOG.md). 48 runs in total.

## Results

Leak-free solutions, out of 6 tasks:

| Assistant | realistic | blind | neutral | review |
|---|---|---|---|---|
| ChatGPT | 4/6 | 5/6 | 6/6 | 6/6 |
| Claude | 3/6 | 6/6 | 6/6 | 6/6 |
| **Both** | **7/12** | **11/12** | **12/12** | **12/12** |

By task (both assistants):

| Task | Leak | realistic | blind | neutral | review |
|---|---|---|---|---|---|
| t01 | Oversampling before the split + duplicate rows | **0/2** | 2/2 | 2/2 | 2/2 |
| t02 | Shuffled time-series CV + look-ahead feature | 2/2 | 2/2 | 2/2 | 2/2 |
| t03 | Same patient in train and test | 1/2 | 2/2 | 2/2 | 2/2 |
| t04 | Feature selection before CV | 2/2 | 2/2 | 2/2 | 2/2 |
| t05 | Model selection on the test set | 2/2 | 2/2 | 2/2 | 2/2 |
| t06 | Test set from the training county only | **0/2** | 1/2 | 2/2 | 2/2 |

## What the results show

**1. Leaks visible in the code were fixed every time.** Shuffled time-series folds (t02), selection before cross-validation (t04) and selecting on the test set (t05) were fixed in every run, in every condition, by both assistants. Both also removed t01's oversampling-before-the-split, the bug written into the starter code, in every run.

**2. Leaks that live in the data were missed without the generator.** t01's data contains retried transactions, exact duplicate rows that a short preview doesn't show. Under `realistic`, neither assistant checked for duplicates, and both left 100 duplicate rows straddling train and test. A comment in Claude's code stated that its test set "has no duplicated rows"; the grader shows otherwise. Once the generator source was visible, both removed the duplicates every time.

**3. Leaks defined by deployment context weren't caught without it.** t06's starter trains and tests on one county, but the model is meant for five. Under `realistic`, neither assistant produced a county-level validation: ChatGPT kept Nyeri only, and Claude pooled all five counties so every county was in both training and validation. Under `blind`, Claude did, after reading the county-only shift in the generator code; ChatGPT described "a strong Nyeri-specific signal in f3" and still kept Nyeri only. With the scenario, both got it right every time. This is arguably a fair outcome: without being told the deployment target, testing on Nyeri is not clearly wrong.

**4. Seeing how the data was made did most of the work.** Adding the generator source (`realistic` → `blind`) moved the pass count from 7/12 to 11/12. Adding the scenario and then an explicit request to review added the last run. The assistants' notes quoted values that exist only in the generator ("+2.5 * label", "SD 3"), so they were reading it, not inferring from data.

**5. Correct ideas can still ship as broken code.** Claude's `realistic` t03 answer split by patient correctly, but crashed on a pandas indexing error (reproduced on pandas 2.2.3 and 3.1.1). One crash in 48 runs.

## What the graders taught us

Five grader or harness problems were found, and **every one came from a correct answer being rejected**:

| Problem | Found by |
|---|---|
| Required the metric to be spelled `pr_auc` (scikit-learn calls it `average_precision`) | ChatGPT, pilot |
| Rejected retried duplicates kept together inside the test set, which don't leak, and compared positive rates on mismatched bases | ChatGPT, pilot |
| Recomputed ROC-AUC per visit only; per-patient AUC is the clinically standard unit | ChatGPT, t03 |
| Accepted a qualifier only before the metric name (`patient_level_roc_auc`), not after it (`roc_auc_patient_level`) | Claude, t03 and t06 |
| A 15-minute limit rejected a correct but compute-heavy answer (~21,000 model fits per run) | ChatGPT, t05 |

Two lessons: **check graders against real model outputs before trusting a single failure**, and **when an answer exposes a grader flaw, fix the rule, not the instance.** The fourth problem was caused by fixing the third too narrowly. Every fix is a regression test, and graders were frozen before the final runs. Details are in the [protocol](PROTOCOL.md).

## Limitations

- **One run per cell.** Assistants are not deterministic, so each pass or fail is one observation. Repeat runs are needed before claiming rates.
- **ChatGPT's model is unknown.** The free tier didn't display a model name, so results can't be tied to a specific model version.
- **Conditions were added during the study.** `blind` and `realistic` were added after earlier runs revealed weaknesses in the design; ChatGPT's `blind` runs came after its other runs. All results were re-graded with the final graders.
- **Web chat, no code execution.** The assistants couldn't run code or inspect the full dataset. An agent that can run `data.duplicated().sum()` might catch t01.
- **Synthetic tasks with public reference solutions.** Real projects are messier, and models may have seen similar examples.
- **Review assisted by a Claude model.** The failure review was done with help from an AI assistant that is itself a Claude model, while Claude was one of the systems evaluated. To limit bias, grading is automatic, every grader change is a general rule with regression tests on both assistants' answers, and every classification with its evidence is public in the [review log](REVIEW_LOG.md).

## Future work

- Repeat the `realistic` runs 3 times per assistant, to see whether the misses are consistent.
- Evaluate coding agents that can run code, which could check the data the way an engineer would.
- Add Gemini and other assistants.
- Extend beyond leakage: unstable training, irreproducibility, performance bottlenecks.

## Reproduce

```bash
make setup
make evaluate   # re-grades all 48 committed answers in evals/runs/
```
