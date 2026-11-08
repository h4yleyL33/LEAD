# PROGRESS — Living Status

**Last updated:** 2026-11-09
**Current milestone:** M3 complete (features + splits + leakage tests green) → M4 modelling

---

## Snapshot

| Phase | Milestone | Status | Evidence |
|---|---|---|---|
| — | M0 Foundation (repo, venv, docs, git) | ✅ Done | `git log`, `.venv`, this file |
| 1 Business Understanding | M1 Framing, KPIs, cost model | ✅ Done | `research/crisp_dm/01_business_understanding.md` |
| 2 Data Understanding | M2 Acquire + audit + EDA | ✅ Done | `results/metrics/data_audit.json`, `eda_summary.json`, 8 figures |
| 3 Data Preparation | M3 Features + splits + leakage tests | ✅ Done | `data/processed/features.parquet`, `splits.json`, 6/6 tests green |
| 4 Modeling | M4 Baselines + XGBoost + tuning | 🔄 Next | — |
| 5 Evaluation | M5 Holdout, calibration, SHAP, errors | ⬜ Pending | — |
| 6 Deployment | M6 FastAPI backend | ⬜ Pending | — |
| 6 Deployment | M7 Interactive dashboard | ⬜ Pending | — |
| 6 Deployment | M8 Geothermal simulation | ⬜ Pending | — |
| — | M9 IEEE manuscript | ⬜ Pending | — |
| — | M10 Reproducibility + final tests | ⬜ Pending | — |

## Verified dataset facts (from `results/metrics/data_audit.json`)

These are measured, not assumed. Regenerate with `python -m src.data.audit`.

- Source: LEAD1.0-small, `github.com/samy101/lead-dataset` (Gulati & Arjunan, BuildSys 2022).
- 1,749,494 rows · 200 buildings · 8,784 hourly timestamps · 2016-01-01 → 2016-12-31.
- 107,653 missing `meter_reading` (6.15%); worst building 61.59% missing.
- 0 duplicate (building, timestamp) pairs; 0 negative readings; 25 zero readings.
- 37,296 anomalies → **2.13% positive rate**, imbalance 45.91:1.
- Per-building anomaly rate: min 0.58%, median 1.94%, max 8.82% (all 200 buildings contain
  at least one anomaly; none is 100% anomalous).
- 5,503 contiguous anomaly events; median 2 h; max 167 h; 47.2% are single-hour events.
- 96 of 200 buildings cover the full hourly grid; 7,306 (building, hour) cells are absent entirely.

## Verified feature/split facts (from `results/metrics/feature_metadata.json`, `splits_summary.csv`)

- 36 causal features in 7 families (level 5, deviation 8, dynamics 9, volatility 4,
  availability 3, calendar 6, peer 1).
- Median autocorrelation 0.933 at lag 1 h, 0.681 at lag 24 h → building-disjoint split mandatory.
- Split 140/30/30 buildings → anomaly rate 2.162% / 2.049% / 2.072% (balanced by construction).
- 5 grouped CV folds of 28 training buildings each.

## Blockers

- **B-001** Kaggle competition split unavailable (no API credentials). Non-blocking for science;
  see `BLOCKERS.md`.

## Next actions (ordered)

1. Baselines (majority, logistic regression, isolation forest) + XGBoost, grouped CV, capped Optuna;
   log every run to `EXPERIMENT_LOG.md`.
2. Holdout evaluation on the 30 test buildings: ROC-AUC/PR-AUC, calibration, threshold analysis,
   event-level recall, SHAP.
3. FastAPI backend → interactive dashboard → geothermal simulation.
4. IEEE manuscript from regenerated artifacts.

## Changelog

- 2026-11-09 — Repo initialised; venv (Python 3.14.9) with numpy 2.5.4, pandas 3.1.1,
  scikit-learn 1.9.1, xgboost 3.4.1, shap 0.54.1, optuna, fastapi. LEAD1.0-small downloaded and
  audited. Planning docs created.
- 2026-11-09 — M2 complete: 8 EDA figures + `02_data_understanding.md`. Key findings: anomaly
  labels are event-like (median 2 h, 47.2% singletons), anomaly rate nearly flat by hour,
  building-scale heterogeneity 57–969 kWh → relative features required.
- 2026-11-09 — M3 complete: `build_features.py` (36 causal features, grid-reindexed per building),
  `splits.py` (140/30/30 building-disjoint, rate-balanced), `test_features_no_leakage.py`
  (6 tests incl. an empirical truncation test). The tests found two real defects: a stateful
  module RNG that made the truncation test compare different data, and a split rebalancer that
  concentrated the highest-anomaly-rate buildings into train (2.50% vs 1.27%); both fixed.
