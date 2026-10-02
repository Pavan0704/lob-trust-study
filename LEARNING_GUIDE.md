# Before using this project in an application

This project was designed and implemented with Codex. Reading a README is not
enough to claim you can defend the work. Allow several focused study sessions.

1. **Market basics:** Explain bid, ask, spread, mid-price and book depth. Explain
   why 10 order-book events are not a fixed wall-clock interval. Read the FI-2010
   paper and the DeepLOB authors' data loader.
2. **Trace one sample:** In `windows`, identify feature rows, label row, first and
   last input event, prediction horizon and output class. Explain why using a future
   label as an input would leak information. Run the unit tests.
3. **Trace the split:** Explain training, checkpoint-selection and calibration
   regions, the purge gaps, and why test data cannot choose a temperature or
   confidence threshold. Explain the unresolved upstream/boundary limitations.
4. **Models:** Explain logistic regression and the CNN tensor shapes. Why might a
   simple model beat the CNN? Why is this not a generative model? Calculate the
   number of CNN parameters with `sum(p.numel() for p in model.parameters())`.
5. **Metrics:** Explain macro-F1 under imbalance; NLL and Brier as probability
   metrics; ECE bin dependence. Calculate each for a three-example toy prediction.
6. **Calibration:** Why does a positive scalar temperature preserve class
   predictions? Why can calibration fitted earlier fail later? Verify raw and
   calibrated F1 agree in recorded results.
7. **Abstention:** Explain why a median threshold on calibration data does not
   guarantee 50% test coverage. Explain why retained accuracy alone is misleading.
8. **Reproduce:** Rerun the documented experiment. Compare your JSON with recorded
   results. Document any differences rather than editing them away.
9. **Own a change:** As a personal learning exercise, add class-wise retained
   coverage or inspect a different horizon. Clearly distinguish your contribution
   from the original AI-assisted implementation and preserve the original results.

Interview check: explain the strongest baseline, one failure, what the experiment
cannot conclude, and what you would investigate next without looking at notes.
