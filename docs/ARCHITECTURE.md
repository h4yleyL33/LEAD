# Architecture

How the pieces fit, in one page. If you are reading the code for the first time, read this first.

## Data flow

```
data/raw/lead/lead1.0-small.csv          1.75 M rows, real LEAD1.0 annotations (git-ignored)
        │
        ├─ src/data/audit.py       ─────► results/metrics/data_audit.json
        ├─ src/data/eda.py         ─────► results/figures/eda_*.png
        │
        ▼
src/features/build_features.py  ───────►  data/processed/features.parquet   (34 causal features)
        │                                 results/metrics/feature_metadata.json
        ▼
src/features/splits.py          ───────►  data/processed/splits.json         (140 / 30 / 30 buildings)
        │
        ├─ src/models/baselines.py      ─► results/metrics/baselines_val.json
        ├─ src/models/tune.py --grid    ─► results/metrics/tuning_best.json
        ├─ src/models/train_xgboost.py  ─► models/xgb_{primary,tuned,weighted}.json
        ├─ src/models/ablation.py       ─► results/metrics/ablation_families_only.json
        ├─ src/models/select_final.py   ─► models/xgb_final.json  (frozen)
        │
        ├─ src/evaluation/holdout.py    ─► results/metrics/holdout_test_final.json
        │                                  results/predictions/test_predictions_final.parquet
        │                                  results/tables/holdout_per_building_final.csv
        ├─ src/evaluation/compare.py    ─► results/tables/model_comparison_{val,test}.{csv,md}
        ├─ src/explainability/shap_explain.py ─► results/metrics/shap_summary_final.json
        │
        └─ src/simulation/geothermal.py ─► data/synthetic/geothermal_signals_SIMULATED.csv
           src/simulation/score_simulated.py ─► results/metrics/simulation_detection_SIMULATED.json
```

## Serving path

```
HTTP ──► backend/app/main.py ──► src/inference/service.py ──► src/features/build_features.py
                │                        │                            (same code as training)
                │                        └──► models/xgb_final.json + threshold from
                │                             results/metrics/xgb_final_val.json
                └──► backend/app/static/index.html (dashboard, canvas, no dependencies)
```

**Invariant:** features are computed by exactly one function. `tests/test_inference.py` scores the
same row through the API path and the stored feature matrix and requires agreement to 1e-6. This is
what makes the "train/serve consistency" claim in the paper checkable rather than aspirational.

## Module responsibilities

| Module | Owns | Does not own |
|---|---|---|
| `src/utils/config.py` | paths, seeds, column names | any logic |
| `src/data/audit.py` | data quality facts | any cleaning |
| `src/data/eda.py` | exploratory figures | any decisions |
| `src/data/load.py` | loading, split slicing, observed-only filter | feature computation |
| `src/features/build_features.py` | the causal feature contract | model choice |
| `src/features/splits.py` | building-disjoint assignment | anything label-dependent beyond balancing |
| `src/evaluation/metrics.py` | metric definitions, threshold selection, bootstrap | model fitting |
| `src/models/*` | training, tuning, selection | evaluation of the test split |
| `src/evaluation/holdout.py` | the single test-set read | model selection |
| `src/inference/service.py` | scoring, alarm typing | feature definitions |
| `backend/app/main.py` | HTTP surface, payload shapes | scoring logic |
| `src/simulation/*` | synthetic signals + transfer test | any claim about a real plant |

## Rules the code enforces

1. **No feature may look forward.** Verified by truncation test, not by review.
2. **No identifier as a feature.** `building_id` never enters the matrix; the protocol is
   building-disjoint.
3. **No cross-asset features.** Every feature must be computable from one asset's history, otherwise
   serving cannot reproduce training (D-012).
4. **No unobserved rows in training or metrics.** A missing reading is never labelled anomalous, so
   including it teaches the wrong rule (D-011).
5. **The test split is read once.** `holdout.py` is run against a frozen `xgb_final.json`.
6. **Numbers flow from script to document.** The paper test compares `paper/main.tex` against
   `results/metrics/*.json` and fails on drift.

## Where to change things

| Goal | Touch |
|---|---|
| Add a feature | `src/features/build_features.py` (then re-run features → training → holdout) |
| Change the split ratio | `src/features/splits.py::SPLIT_SIZES` (invalidates all reported numbers) |
| Change the metric set | `src/evaluation/metrics.py` (then re-run holdout and update the paper) |
| Change the alarm policy | `src/inference/service.py` (threshold, OUTAGE handling) |
| Add an endpoint | `backend/app/main.py` + `backend/tests/test_api.py` |
| Change the simulator | `src/simulation/geothermal.py` (keep every output tagged SIMULATED) |
