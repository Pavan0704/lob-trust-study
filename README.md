# LOB Trust Study

An **AI-assisted introductory portfolio study** of short-horizon limit-order-book
classification, confidence calibration and selective prediction on FI-2010.
Prepared for Pavan Gupta's transition from robust ML evaluation to financial-data
research. Codex designed and implemented the experiment and documentation, and
executed the recorded runs. Pavan must review the code, reproduce the results and
understand the learning guide before presenting this as personal experience.
This is not independently authored applicant research or a published contribution.

## Question

Does a small temporal CNN beat a last-snapshot logistic classifier, and does
calibration learned on an earlier held-out region remain useful on later test
files? What happens if a fixed confidence threshold is used to abstain?
The outcome is allowed to be negative. No model is advertised as superior in advance.

## Data and attribution

- FI-2010, introduced by Ntakaris et al., *Benchmark dataset for mid-price
  forecasting of limit order book data with machine learning methods*, Journal of
  Forecasting (2018), DOI: https://doi.org/10.1002/for.2543.
- Archive and feature/label layout sourced from the DeepLOB authors:
  https://github.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books
- Format follows their `jupyter_pytorch/run_train_pytorch.ipynb`: first 40 rows
  are book features; last five are classification labels. We use the first label
  horizon, 10 book events, not ten seconds. Labels 1/2/3 become 0/1/2.
- Only supplied decimal-preprocessed CF_7 training and CF_7/8/9 test files are used.
  Archive SHA-256 is checked by download_data.py and recorded in results.json.
  Data are excluded from Git. Availability in a research repository does not grant
  a new redistribution license; consult the dataset owners' terms for other uses.

## Protocol

1. Split supplied training columns into 60% training, 20% checkpoint selection,
   and 20% calibration. Remove 100 events around both internal boundaries.
2. Use trailing 20-event windows spaced 40 events apart. Never shuffle columns
   before constructing windows. Training batches may be shuffled afterward.
3. Fit feature mean and standard deviation on training features only; clip scaled
   features to [-10,10]. Labels are never used as input features.
4. Compare training-class-prior and last-snapshot logistic regression baselines
   against a two-layer temporal CNN, 12 epochs, seeds 7/17/27.
   Preserve this initial run, including any collapse to majority-class prediction.
5. Select CNN checkpoint by validation NLL. Fit a scalar temperature by calibration
   NLL on a fixed grid [0.25,8]. Choose confidence threshold from the calibration
   median; do not force test coverage to 50%.
6. Evaluate every test file separately: macro-F1, accuracy, NLL, multiclass Brier,
   ten-bin ECE, confusion matrix, retained coverage and retained accuracy.
7. Mask volume feature slots to zero in standardized space for a controlled stress
   test (training-mean replacement). This is not an observed market regime change.

All three runs use identical data and protocol. Temperature cannot change argmax
predictions, so calibration is assessed through probability metrics, not F1 gains.
After inspecting the first experiment, we added one exploratory class-weighted
follow-up (30 epochs, same seeds and architecture). Weights use training counts
only. This follow-up was motivated by observed majority-class collapse and is
explicitly post hoc: the test files are no longer untouched confirmatory evidence
for choosing between experiments. No further configuration search was performed.

## Run

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python download_data.py
python -m unittest -v test_study
python -u study.py
python -u study.py --balanced --epochs 30
```

Recorded package versions, dataset shapes, class counts, seeds and runtime are in
each experiment's `results.json`.
Reports: [initial](results/initial/REPORT.md) and
[weighted follow-up](results/weighted_followup/REPORT.md).
Per-example labels/logits/probabilities are saved in compressed NPZ files.
No trained weights are published; the complete experiment retrains on CPU.
The recorded run used a project-local environment with NumPy 1.26.4 to support
the available PyTorch 2.0.1 CPU installation. Unrelated inherited packages were
not used in this study; a clean environment is also supported by requirements.txt.

## Limits

- Compact study on a historical benchmark, not a DeepLOB reproduction or a full
  benchmark comparison. The small CNN is deliberately simpler than DeepLOB.
- No raw exchange timestamps, stock identifiers or stock/day boundary metadata are
  reconstructed here. A window may cross an internal concatenation boundary in a
  source file. Internal purges do not solve this limitation. Results should not be
  called stock-specific, day-specific or independent-regime estimates.
- Supplied preprocessing and labels are inherited, not reconstructed from raw
  messages. Our own scaling uses training data only, but upstream leakage has not
  been independently audited. This prevents an end-to-end leakage-free claim.
- Sparse windows reduce overlap; autocorrelation remains. Seed SD is model
  initialization variability, not a confidence interval over independent markets.
- Selection can improve accuracy by discarding difficult examples or a class.
  Coverage must be reported with retained accuracy. ECE depends on binning.
- Synthetic volume masking is only a sensitivity check. Price direction accuracy
  does not establish profitability, execution feasibility or financial robustness.
- No diffusion model, generative financial synthesis, world model, novel algorithm,
  transaction-cost backtest, or live trading is implemented.

## What the recorded experiments actually found

The initial CNN predicts the majority class on all test examples. Its accuracy
therefore cannot be used as evidence of learned directional forecasting. Macro-F1
is 0.282/0.277/0.262 on CF_7/8/9, versus logistic regression's 0.331/0.342/0.279.
The class-weighted exploratory CNN achieves mean macro-F1 0.282/0.289/0.284.
It remains below logistic regression on two files; the small advantage on CF_9 is
not demonstrated statistically significant. The weighted experiment also changes
the epoch budget, so it does not isolate class weighting as the cause of a change.

Calibration lowers NLL on the first two files for the initial CNN, but not
meaningfully on the third. All weighted-run temperatures reach the grid's upper
boundary of 8. This signals that the scalar correction is insufficient within the
chosen range. Weighted calibrated NLL remains around 1.10; compare the class-prior
baseline in JSON before describing calibration as successful. Weighted-run
selective accuracy is only 0.318/0.334/0.285 at roughly 57% coverage. Abstention
does not rescue these models. These failures are intentionally retained.

This supports a lesson about imbalance, baseline comparisons and probability
evaluation, not a claim of strong financial forecasting performance.

## Application wording

Use only after studying and reproducing the project:

> AI-assisted learning project in financial ML: evaluated a small temporal CNN
> against simple baselines on FI-2010 limit-order-book data, with separated model
> selection/calibration, probability metrics and confidence-based abstention.

Do not claim independent authorship, expertise in quantitative finance, state-of-
the-art results or a new research method. Cite actual measured results only.

## Next meaningful work

Recover stock/day boundaries and audit upstream normalization; compare more
forecast horizons and stronger baselines; assess class-wise selective coverage;
then investigate generative sequence modeling with economic and temporal fidelity
checks. These are future directions, not completed features.
