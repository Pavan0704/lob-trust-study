# Recorded results

AI-assisted introductory study. All numbers computed by study.py.

| Test file | Prior F1 | Logistic F1 | CNN F1, mean ± SD | Raw NLL | Calibrated NLL | Retained coverage | Retained accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| CF_7 | 0.282 | 0.331 | 0.282 ± 0.003 | 1.257 | 1.107 | 0.575 | 0.318 |
| CF_8 | 0.277 | 0.342 | 0.289 ± 0.011 | 1.220 | 1.103 | 0.574 | 0.334 |
| CF_9 | 0.262 | 0.279 | 0.284 ± 0.004 | 1.291 | 1.111 | 0.577 | 0.285 |

SD is across initialization seeds, not a statistical confidence interval.
Temperature and abstention threshold are fitted on calibration data only.
File names are the source convention; these are not independently identified market regimes.
See results.json for all metrics, confusion matrices, stress checks and training histories.
