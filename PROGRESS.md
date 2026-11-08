# PROGRESS — Living Status

**Last updated:** 2026-11-09
**Current milestone:** M4 + M5 complete (model trained, selected, evaluated once on held-out buildings) → M6 backend

---

## Snapshot

| Phase | Milestone | Status | Evidence |
|---|---|---|---|
| — | M0 Foundation (repo, venv, docs, git) | ✅ Done | `git log`, `.venv`, this file |
| 1 Business Understanding | M1 Framing, KPIs, cost model | ✅ Done | `research/crisp_dm/01_business_understanding.md` |
| 2 Data Understanding | M2 Acquire + audit + EDA | ✅ Done | `results/metrics/data_audit.json`, `eda_summary.json`, 8 figures |
| 3 Data Preparation | M3 Features + splits + leakage tests | ✅ Done | `data/processed/features.parquet`, `splits.json`, 6/6 tests green |
| 4 Modeling | M4 Baselines + XGBoost + tuning | ✅ Done | `results/tables/model_comparison_val.csv`, `models/xgb_final.json` |
| 5 Evaluation | M5 Holdout, calibration, SHAP, errors | ✅ Done (SHAP pending in M6/M9) | `results/metrics/holdout_test_final.json`, 4 figures |
| 6 Deployment | M6 FastAPI backend | 🔄 Next | `src/inference/service.py` ready |
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

## Verified model results (from `results/metrics/`)

Validation (30 buildings, 250,019 observed rows) — model selection only:

| Model | ROC-AUC | PR-AUC | Brier | event recall |
|---|---|---|---|---|
| prior | 0.5000 | 0.0215 | 0.02104 | — |
| logistic | 0.8087 | 0.5715 | 0.01304 | 0.7803 |
| isolation forest | 0.8058 | 0.2296 | 0.13484 | 0.4740 |
| XGBoost `tuned` (selected) | 0.9562 | 0.8198 | 0.00662 | 0.8962 |

Held-out test (30 unseen buildings, 243,684 rows, 5,454 anomalies) — opened once:

- ROC-AUC **0.9524** (95% CI 0.9249–0.9702, building-level bootstrap)
- PR-AUC **0.8342** (base rate 0.0224), Brier 0.00613, ECE 0.0005
- precision/recall 0.7987/0.7888 at threshold 0.1283; alert rate 2.21%
- event recall 0.8228 (441/536 events), median latency 1 h, 0.056 false alarms per building-day
- median per-building recall 0.842; 4 of 30 buildings below 0.5 (worst: 0.017)

## Blockers

- **B-001** Kaggle competition split unavailable (no API credentials). Non-blocking for science;
  see `BLOCKERS.md`.

## Next actions (ordered)

1. FastAPI backend (`backend/app`) exposing `/health`, `/model`, `/score`, `/batch`, `/simulate`.
2. Interactive dashboard (single-page, served by the backend) over the scored test predictions.
3. Geothermal equipment simulation (`src/simulation`, labelled SIMULATED throughout).
4. SHAP explanations + IEEE manuscript from regenerated artifacts.
5. `pytest` suite green end-to-end + reproducibility check (`make`-style script).

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
- 2026-11-09 — M4/M5 complete. Baselines (logistic 0.809, iforest 0.806) → XGBoost selected at
  0.9562 val ROC-AUC / 0.8198 PR-AUC. Held-out test (single read): ROC-AUC 0.9524
  (0.9249–0.9702), PR-AUC 0.8342, event recall 0.8228 at 1 h median latency. Two integrity issues
  found and fixed mid-flight: (a) missing readings are never labelled → observed-only training
  (D-011); (b) a cross-building peer feature broke train/serve consistency by 57% → removed
  (D-012). A TPE study was invalidated by seeded-RNG replay and replaced by a 6-candidate grid.
