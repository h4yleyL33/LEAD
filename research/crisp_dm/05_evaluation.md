# Phase 5 — Evaluation

**CRISP-DM phase:** 5 of 6
**Status:** complete (single test-set read)
**Reproduce:**
```bash
python -m src.evaluation.holdout --variant final     # metrics, predictions, 4 figures
python -m src.evaluation.compare                     # comparison tables
```

The test split was opened **once**, after the model was frozen (`models/xgb_final.json`) and its
threshold chosen on validation (0.1283). Nothing below was re-tuned after seeing these numbers.

---

## 5.1 Headline result — 30 held-out buildings, 243,684 observed hours, 5,454 anomalies

| Metric | Value |
|---|---|
| ROC-AUC | **0.9524** (95% CI 0.9249–0.9702, building-level bootstrap, 300 resamples) |
| PR-AUC | **0.8342** (base rate 0.0224 → 37× lift) |
| Brier | 0.00613 |
| ECE (12 quantile bins) | **0.0005** |
| Recall @ FPR = 1% | 0.8089 |
| Recall @ FPR = 0.1% | 0.7329 |
| At threshold 0.1283: precision / recall / F1 | 0.7987 / 0.7888 / 0.7937 |
| Alert rate | 2.21% of hours |
| Anomaly **events** (n) | 536 |
| Events with ≥1 hour flagged | 441 → **event recall 0.8228** |
| Median detection latency | **1 hour** |
| False-alarm episodes per building-day | 0.056 |

The confidence interval is wide (±0.023) because it resamples *buildings*, not rows — 30 clusters is
what the protocol gives, and a row bootstrap would have produced a falsely narrow interval.

## 5.2 Point ranking vs episode detection

The two metric families disagree in an instructive way:

- Point-wise, 78.9% of anomalous hours are caught at the chosen operating point.
- Episode-wise, 82.3% of anomalies are detected at all, with a median latency of one hour.

The gap runs the *good* direction here (episodes are easier than individual hours, because an
episode only needs one flagged hour), and it is the number a maintenance team cares about: "did we
raise an alarm for this fault, and how long after it started?" Both are reported because a model can
score well on one and badly on the other — the family ablation in §4.6 shows level-only features at
0.90 ROC-AUC but 0.43 event recall.

## 5.3 Calibration

ECE is 0.0005 over 12 quantile bins spanning predicted probabilities from ~3×10⁻⁴ to ~0.3
(`results/figures/holdout_calibration_final.png`). Predicted probabilities are directly usable:
when the model says 0.30, roughly 30% of such hours are anomalous.

This is a consequence of *not* reweighting the classes (§4.5). The weighted variant has 3.8× the
Brier score and would require post-hoc recalibration before its probabilities could be shown to an
operator.

**Measurement note.** The first version of this figure used `sklearn.calibration.calibration_curve`
with 15 quantile bins and produced a two-point curve, because repeated quantile edges collapse in a
2%-prevalence score distribution. It looked like perfect calibration and was meaningless. The
current implementation bins explicitly, keeps the bin counts, and plots on log axes
(`src/evaluation/metrics.py::reliability_curve`); the figure now shows 12 populated bins and their
sizes.

## 5.4 Per-building behaviour (the honest part)

`results/tables/holdout_per_building_final.csv`:

| Statistic | Value |
|---|---|
| Median per-building recall | 0.842 |
| Buildings with recall > 0.9 | 8 of 30 |
| Buildings with recall < 0.5 | 4 of 30 |
| Worst building | id 32: 58 anomalies, 1 flagged (recall 0.017), mean score 0.009 |

The worst cases are buildings whose annotated anomalies are *small relative to their own
volatility*. Because every feature is normalised by the building's own recent history (D-012,
relative features only), a 20% excursion in a noisy building looks exactly like normal operation.
This is an intrinsic property of self-referential anomaly detection, not a bug, and it defines the
deployment caveat: the model detects deviations from a building's *learned* normal, so a building
that is chronically abnormal will be under-flagged.

The distribution is bimodal rather than uniform — 8 buildings above 0.9, 4 below 0.5 — which means
an aggregate AUC hides real heterogeneity. Any deployment should report per-asset recall, which is
why the per-building table is a first-class artefact and not an appendix.

## 5.5 What the model has actually learned

Gain-based importance on the final model:

| Rank | Feature | Gain % | Reading |
|---|---|---|---|
| 1 | `f_dev_rel_720` | 12.7 | current reading vs its own 30-day mean |
| 2 | `f_abs_diff1` | 12.3 | size of the hour-to-hour jump |
| 3 | `f_dev_rel_168` | 8.5 | current reading vs its own 7-day mean |
| 4 | `f_log_value` | 6.3 | absolute level |
| 5 | `f_ratio_med_168` | 5.5 | current / 7-day median |
| 6 | `f_diff1_z24` | 4.6 | jump size relative to that building's typical jumpiness |
| 7 | `f_z_6` | 4.5 | deviation from the last 6 hours in std units |
| 8 | `f_range_24` | 3.7 | recent spread before the point |

This is physically sensible for metered demand: anomalies in LEAD1.0 are mostly *step changes and
spikes* relative to a building's own baseline, not absolute magnitude. It also explains why the
`dynamics` family alone reaches 0.928 ROC-AUC.

## 5.6 Threats to validity

| Threat | Status |
|---|---|
| Label noise | LEAD1.0 annotations are human/rule-assisted; some "anomalies" are benign operating changes. Caps achievable AUC; not quantified here because no second annotator exists. |
| One dataset, one year, one climate | All 200 buildings are from the ASHRAE GEP III pool (2016). Transfer to a geothermal plant is untested — the geothermal module is explicitly **simulated** (D-005). |
| Single train/test draw | The split is deterministic and fixed; no repeated-split variance is reported. The building-level bootstrap CI partly covers this. |
| Threshold transfer | The F2-optimal threshold (0.1283) is calibrated to a 2.2% prevalence. At a plant with a different anomaly rate, the threshold must be re-derived — the API exposes it as configuration rather than hard-coding it. |
| Hyper-parameter selection | Flat surface (§4.3); the choice of `tuned` over `primary` is within noise and does not affect the conclusions. |
| Leakage | Addressed by construction and by test (`tests/test_features_no_leakage.py`, empirical truncation test), plus the observed-readings-only rule (D-011). |

## 5.7 Comparison with published LEAD1.0 results — deliberately not made

Published LEAD1.0 benchmarks use the Kaggle competition split and metric definitions that this
environment cannot reproduce (blocker B-001: no credentials, no mirror). Quoting another paper's
number next to ours would compare different splits, different label sets and possibly different
metrics. The paper therefore reports only internally consistent numbers and states the omission
explicitly.

## 5.8 Figures

| File | Contents |
|---|---|
| `holdout_roc_final.png` | ROC on held-out buildings, AUC annotated |
| `holdout_pr_final.png` | Precision–recall with base-rate reference |
| `holdout_calibration_final.png` | Reliability curve (log-log) + bin counts |
| `holdout_scores_final.png` | Score density by class with the operating threshold |

## 5.9 Exit criteria (met)

- [x] Test set evaluated once, with the frozen model and frozen threshold.
- [x] Point, event and calibration metrics reported together.
- [x] Uncertainty quantified with a building-level bootstrap.
- [x] Per-building heterogeneity reported, worst cases named.
- [x] Threats to validity written down before the manuscript.
- [x] No comparison with incomparable published numbers.
