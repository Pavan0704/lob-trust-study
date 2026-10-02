# Recorded results

AI-assisted introductory study. All numbers computed by study.py.

| Test file | Prior F1 | Logistic F1 | CNN F1, mean ± SD | Raw NLL | Calibrated NLL | Retained coverage | Retained accuracy |
|---|---:|---:|---:|---:|---:|---:|---:|
| CF_7 | 0.282 | 0.331 | 0.282 ± 0.000 | 0.816 | 0.784 | 0.789 | 0.755 |
| CF_8 | 0.277 | 0.342 | 0.277 ± 0.000 | 0.836 | 0.813 | 0.753 | 0.737 |
| CF_9 | 0.262 | 0.279 | 0.262 ± 0.000 | 0.905 | 0.905 | 0.770 | 0.679 |

SD is across initialization seeds, not a statistical confidence interval.
Temperature and abstention threshold are fitted on calibration data only.
File names are the source convention; these are not independently identified market regimes.
See results.json for all metrics, confusion matrices, stress checks and training histories.
