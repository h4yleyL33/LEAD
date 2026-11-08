# Phase 3 — Data Preparation

**CRISP-DM phase:** 3 of 6
**Status:** complete (feature set v1, frozen for the first modelling round)
**Reproduce:**
```bash
python -m src.features.build_features   # -> data/processed/features.parquet + results/metrics/feature_metadata.json
python -m src.features.splits           # -> data/processed/splits.json + splits_summary.csv
python -m pytest tests/test_features_no_leakage.py -q
```

---

## 3.1 The feature contract

> Every feature value at time *t* is a deterministic function of
> `{(building b, reading at time s) : s ≤ t}` and of nothing else.

This single sentence is the whole design constraint, and it is enforced by test rather than by
review (`tests/test_features_no_leakage.py`, 6 tests, all passing):

| Test | What it would catch |
|---|---|
| `test_no_feature_uses_future_information` | Build features on full history and on history truncated at *T*; every value at *t ≤ T* must be identical. Catches centred windows, negative shifts, `bfill`, and whole-series normalisers. |
| `test_baseline_excludes_current_reading` | Corrupt only the reading at *t*; pure-past features (`f_cv_*`, `f_range_*`, `f_momentum_*`) must not move. |
| `test_ratio_denominator_is_past_only` | For `value / baseline` features, the implied baseline must not move with the current reading. |
| `test_building_id_is_not_a_feature` | The identifier is excluded — the protocol is building-disjoint, so it cannot generalise. |
| `test_split_is_building_disjoint` | No building may appear in two splits; the grouped folds must cover exactly the training buildings. |
| `test_splits_are_deterministic` | Re-running the splitter must produce byte-identical splits. |

Two engineering details that make the contract hold in practice:

1. **Each building is reindexed onto the full hourly grid** of the dataset before rolling. Without
   this, a 24-row window would silently span 30+ hours wherever rows are absent, and the feature
   would mean something different per building. Absent rows are *not* re-emitted: features exist
   only for rows present in the source data.
2. **Past-only baselines are computed on `series.shift(1)`**, so the current reading never enters
   its own baseline. `min_periods` is set per window (6 h → 3, 24 h → 12, 168 h → 24, 720 h → 48)
   so that short history yields NaN rather than a confident estimate from two points.

## 3.2 Feature families (36 features)

| Family | n | Members | Rationale |
|---|---|---|---|
| `level` | 5 | `f_log_value`, `f_ratio_med_{6,24,168,720}` | Scale-free level. Raw kWh does not transfer between buildings (medians span 57–969 kWh), so level enters only as `log1p` or as a ratio to the building's own recent median. |
| `deviation` | 8 | `f_z_{6,24,168,720}`, `f_dev_rel_{6,24,168,720}` | How far the current point sits from its own recent distribution, in std units and relative units. |
| `dynamics` | 9 | `f_diff1`, `f_abs_diff1`, `f_diff1_z24`, `f_delta_24`, `f_delta_168`, `f_ratio_24`, `f_ratio_168`, `f_momentum_6_24`, `f_momentum_24_168` | Same-hour-yesterday / last-week comparisons and short-vs-long momentum — the signatures of a step change versus normal cycling. |
| `volatility` | 4 | `f_cv_{24,168}`, `f_range_{24,168}` | Recent dispersion *before* the current point: a burst is anomalous in a quiet building, unremarkable in a noisy one. |
| `availability` | 3 | `f_is_missing`, `f_missing_24h`, `f_hours_since_valid` | 6.15% of readings are missing; dropouts cluster around asset trouble, and a frozen sensor looks "flat", not "high". |
| `calendar` | 6 | `f_hour_sin/cos`, `f_dow_sin/cos`, `f_month`, `f_is_weekend` | Cyclical encoding; anomaly rate varies 1.10%–3.61% across weekday × hour. |
| `peer` | 1 | `f_peer_dev` | Contemporaneous cross-building deviation of the 24 h ratio (scale-free). Uses only values observable at *t*. |

NaN rates are 0% (calendar, availability flags) to 9.7% (`f_delta_168` — buildings with less than a
week of history). XGBoost consumes NaN natively; no imputation is applied, so the missingness
pattern stays visible to the model.

## 3.3 What is deliberately *not* a feature

- `building_id` — unlearnable for unseen buildings under the chosen protocol.
- Any lag/lead with a negative offset, any centred rolling window, any `bfill`.
- Any target-derived statistic (target encoding, "anomaly count in the last 24 h"). These are the
  single easiest way to reach a fake 0.99 AUC on this dataset.
- Raw `meter_reading` in kWh, for the scale reasons above.

## 3.4 Split protocol (fixed before any model was trained)

Buildings are sorted by per-building anomaly rate and assigned greedily to whichever split is
furthest behind its target share (140 / 30 / 30). Sorting before assignment — rather than shuffling —
is what keeps the label rate balanced across splits.

| Split | Buildings | Rows | Anomalies | Anomaly rate | Missing rate |
|---|---|---|---|---|---|
| train | 140 | 1,223,948 | 26,466 | 2.162% | 6.19% |
| val | 30 | 262,361 | 5,376 | 2.049% | 4.70% |
| test | 30 | 263,185 | 5,454 | 2.072% | 7.41% |

`train` is additionally partitioned into 5 grouped folds (28 buildings each) for hyper-parameter
selection; `val` is used for early stopping and threshold-free model comparison; `test` is touched
once, at the end, and its numbers are what the paper reports.

## 3.5 Why this protocol, in one paragraph

Median lag-1 autocorrelation of `meter_reading` is 0.933. Under a random row split, a model can
predict an anomaly at 14:00 by reading 15:00, and the reported AUC would be meaningless. Under a
random *time* split (train on Jan–Sep, test on Oct–Dec) the same building's idiosyncrasies leak
through the building-level normalisation. Only a building-disjoint split answers the question the
business actually asks — *will this catch faults in a plant we have not instrumented before?* —
and it is the harder, more honest number.

## 3.6 Artefacts

| Path | Contents |
|---|---|
| `data/processed/features.parquet` | 1,749,494 × 38 (keys + 36 features + label), 164 MB, zstd |
| `data/processed/splits.json` | building ids per split + 5 grouped folds + seed |
| `data/processed/splits_summary.csv` | the table in §3.4 |
| `results/metrics/feature_metadata.json` | feature list, family tags, NaN rates, window definitions |

## 3.7 Exit criteria (met)

- [x] Feature set built from raw data by a re-runnable script.
- [x] Causality enforced by an executable truncation test, not by assertion.
- [x] Split protocol fixed and written down before training.
- [x] No identifier, no target-derived feature, no raw level.
- [x] 6/6 leakage and split tests passing.
