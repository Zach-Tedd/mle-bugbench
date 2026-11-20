| Task | chatgpt (realistic) | chatgpt (blind) | chatgpt (neutral) | chatgpt (review) | claude (realistic) | claude (blind) | claude (neutral) | claude (review) |
|---|---|---|---|---|---|---|---|---|
| t01_oversample_leak | fail (20) | pass (100) | pass (100) | pass (100) | fail (50) | pass (100) | pass (100) | pass (100) |
| t02_timeseries_cv_leak | pass (100) | pass (100) | pass (100) | pass (100) | pass (100) | pass (100) | pass (100) | pass (100) |
| t03_patient_leak | pass (100) | pass (100) | pass (100) | pass (100) | error | pass (100) | pass (100) | pass (100) |
| t04_feature_selection_leak | pass (100) | pass (100) | pass (100) | pass (100) | pass (100) | pass (100) | pass (100) | pass (100) |
| t05_test_set_selection | pass (100) | pass (100) | pass (100) | pass (100) | pass (100) | pass (100) | pass (100) | pass (100) |
| t06_regional_shift | fail (50) | fail (50) | pass (100) | pass (100) | fail (70) | pass (100) | pass (100) | pass (100) |

| Assistant | Condition | Passed | Mean score |
|---|---|---|---|
| chatgpt | realistic | 4/6 | 78 |
| chatgpt | blind | 5/6 | 92 |
| chatgpt | neutral | 6/6 | 100 |
| chatgpt | review | 6/6 | 100 |
| claude | realistic | 3/6 | 70 |
| claude | blind | 6/6 | 100 |
| claude | neutral | 6/6 | 100 |
| claude | review | 6/6 | 100 |
