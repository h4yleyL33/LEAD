# DECISIONS — Architecture & Research Decision Log

Format: ADR-lite. Each entry: context → decision → consequences. Newest last.

---

## D-001 — Use LEAD1.0-small as the primary real dataset
**Date:** 2026-11-09
**Context:** The specification targets the Kaggle *Large-scale Energy Anomaly Detection* (LEAD)
competition. The competition split (`train.csv` 200 buildings / `test.csv` 206 buildings, 1.34 GB)
requires an authenticated Kaggle account that has accepted the rules; no credentials exist in this
environment (API returns HTTP 401).
**Decision:** Use `lead1.0-small.csv` published by the dataset authors in `samy101/lead-dataset`
(real ASHRAE-derived hourly meter readings with human anomaly annotations, 200 buildings, 1.75 M
rows). Do not fabricate, and do not substitute synthetic data for the model's training labels.
**Consequences:** (+) Real labels, real science, fully reproducible without credentials.
(−) The exact competition test set and `row_id` submission format are unavailable, so no
leaderboard-comparable score can ever be reported. Documented in `BLOCKERS.md` B-001 and in the
manuscript's limitations.

## D-002 — Building-disjoint evaluation instead of random row splits
**Date:** 2026-11-09
**Context:** Hourly meter data is strongly autocorrelated within a building. A random row split
places near-duplicate hours of the same building in both train and test, inflating AUC.
**Decision:** Split by `building_id` (140/30/30, stratified by per-building anomaly rate, seed 42).
**Consequences:** (+) Estimates generalisation to *unseen buildings*, which is the deployment
scenario. (−) Lower, more honest AUC than row-wise splits; the gap is itself reported.

## D-003 — Causal (backward-looking only) feature engineering
**Date:** 2026-11-09
**Context:** Anomaly detection in production must decide at time *t* using data available at *t*.
Centred rolling windows and future-looking lags leak information and are unusable online.
**Decision:** All rolling statistics use `closed='left'`-style past-only windows shifted by one
hour where the current value would otherwise enter the statistic. Peer/building-level statistics
are computed from the training partition only.
**Consequences:** (+) Deployable model, no optimistic bias. (−) Slightly weaker features than
leaky alternatives — by design. Enforced by `tests/test_features_no_leakage.py`.

## D-004 — ROC-AUC as primary metric, PR-AUC and recall@FPR as secondary
**Date:** 2026-11-09
**Context:** 2.13% positive rate. Accuracy is meaningless (97.87% by predicting "normal").
**Decision:** Primary = ROC-AUC; secondary = average precision (PR-AUC), precision/recall/F1 at a
validation-selected threshold, Brier score, recall at FPR ≤ 1%.
**Consequences:** Threshold-free comparison between models, plus an operating-point view for the
maintenance use case.

## D-005 — Geothermal data is simulated and labelled as such
**Date:** 2026-11-09
**Context:** No PLTP Kamojang SCADA export is available (and would be confidential). The spec
requires a geothermal equipment simulation module.
**Decision:** `src/simulation/` generates physically motivated auxiliary-equipment signals
(cooling-water pump, NCG extraction, turbine bearing temperature, separator level) with documented
parameters. All outputs carry a `simulated=True` flag, filenames carry `_SIMULATED`, figures carry
a SIMULATED watermark.
**Consequences:** (+) Demonstrates transferability and gives the dashboard a live data source.
(−) Zero evidential value about the real plant; stated explicitly everywhere.

## D-006 — Python 3.14 + xgboost 3.4.1 (no downgrade needed)
**Date:** 2026-11-09
**Context:** Risk R5 predicted missing wheels for Python 3.14.
**Decision:** Verified empirically that numpy 2.5.4, pandas 3.1.1, scikit-learn 1.9.1,
xgboost 3.4.1, shap 0.54.1, optuna, fastapi all install and import on 3.14.9. Keep the system
interpreter; no 3.12 fallback.
**Consequences:** Single toolchain; `requirements.txt` pins exact versions.

## D-007 — No imputation for the tree model, median imputation for linear baselines
**Date:** 2026-11-09
**Context:** 6.15% of `meter_reading` values are missing, and the missingness itself may be
informative (sensor outage).
**Decision:** XGBoost consumes NaN natively. Linear/logistic baselines use train-fitted median
imputation. A binary `is_missing` indicator is available as a feature.
**Consequences:** Fair comparison without discarding 6% of the data; missingness is modelled
explicitly rather than silently dropped.

## D-008 — Causal feature contract enforced by an executable truncation test
**Date:** 2026-11-09
**Context:** Median lag-1 autocorrelation is 0.933, so any accidental forward-looking feature would
inflate AUC to a meaningless level without being visible in code review.
**Decision:** Features must be deterministic functions of readings at times ≤ t. This is verified by
rebuilding the feature matrix on history truncated at T and asserting bit-identity of every value at
t ≤ T (`tests/test_features_no_leakage.py`). Baselines are computed on `series.shift(1)`.
**Consequences:** (+) Leakage becomes a test failure, not a reviewer opinion. (−) Roughly 30 s of
extra test runtime; no centred smoothing is available even where it would be convenient.

## D-009 — Grid reindexing per building before rolling windows
**Date:** 2026-11-09
**Context:** 7,306 (building, hour) cells are absent and 6.15% of readings are NaN, so row-based
rolling windows would span variable amounts of wall-clock time.
**Decision:** Reindex each building onto the dataset's full hourly grid, compute rolling statistics
there, then emit features only for rows that exist in the source data.
**Consequences:** (+) A 24 h window always means 24 h. (−) Absent hours must not be re-emitted as
training rows; enforced by a row-count assertion in `build_features.main()`.

## D-010 — 140 / 30 / 30 building split, balanced by anomaly rate
**Date:** 2026-11-09
**Context:** Building-disjoint splitting (D-002) with an unlucky assignment would leave the test set
with a materially different anomaly rate than train, making the holdout number hard to interpret.
**Decision:** Sort buildings by per-building anomaly rate and greedily assign each to whichever split
is furthest behind its target share (140/30/30). Train is further partitioned into 5 grouped folds
(28 buildings each) for tuning.
**Consequences:** (+) Anomaly rate 2.162% / 2.049% / 2.072% across splits. (−) Assignment depends on
the anomaly rate, which uses labels — acceptable because the split is fixed before modelling and is
never re-drawn per experiment.

## D-011 — Train and score only on observed readings; outages are a separate alarm class
**Date:** 2026-11-09
**Context:** In LEAD1.0, 0 of the 107,653 hours without a `meter_reading` carry a positive label — an
absent reading cannot be annotated. A model trained on all rows therefore learns the deterministic
rule "missing ⇒ normal", which is both a 6.15% free lunch on every metric and operationally wrong: a
dead sensor is a fault. Measured impact of removing the rule: ROC-AUC 0.9576 → 0.9554.
**Decision:** Drop rows with no reading from training and from all reported metrics; the API reports
such hours as `OUTAGE`, a distinct alarm type, instead of scoring them.
**Consequences:** (+) Metrics describe genuine signal detection; the model is never taught that a
silent asset is healthy. (−) The reported anomaly rate (2.27% on observed rows) differs from the
dataset-wide 2.13%, and the two must not be mixed in the paper.

## D-012 — No cross-building "peer" feature
**Date:** 2026-11-09
**Context:** A cross-sectional feature (`f_peer_dev`, deviation from the median building at the same
hour) is legitimate at training time but not reproducible when scoring one building alone, so the
serving path produced 0.0339 where the batch path produced 0.0216 — a 57% discrepancy caught by
`tests/test_inference.py`.
**Decision:** Remove the peer feature. Every feature must be computable from a single building's own
history. `build_feature_frame(..., include_peer=True)` retains it for reproducing the ablation.
**Consequences:** (+) Exact train/serve agreement (verified to 1e-6) and a deployment that needs no
cross-building infrastructure. (−) The family-only measurement (ROC-AUC 0.7798) is recorded but
unused; removing it did not hurt — validation ROC-AUC 0.9539 → 0.9543, PR-AUC 0.8200 → 0.8239.
