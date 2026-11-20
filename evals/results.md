| Task | chatgpt (blind) | chatgpt (neutral) | chatgpt (review) |
|---|---|---|---|
| t01_oversample_leak | pass (100) | pass (100) | pass (100) |
| t02_timeseries_cv_leak | pass (100) | pass (100) | pass (100) |
| t03_patient_leak | pass (100) | pass (100) | pass (100) |
| t04_feature_selection_leak | pass (100) | pass (100) | pass (100) |
| t05_test_set_selection | pass (100) | pass (100) | pass (100) |
| t06_regional_shift | fail (50) | pass (100) | pass (100) |

| Assistant | Condition | Passed | Mean score |
|---|---|---|---|
| chatgpt | blind | 5/6 | 92 |
| chatgpt | neutral | 6/6 | 100 |
| chatgpt | review | 6/6 | 100 |
