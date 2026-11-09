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
| 6 Deployment | M6 FastAPI backend | ✅ Done | `backend/app/main.py`, 13 API tests |
| 6 Deployment | M7 Interactive dashboard | ✅ Done | `backend/app/static/index.html`, verified in browser |
| 6 Deployment | M8 Geothermal simulation | ✅ Done | `src/simulation/`, `results/metrics/simulation_detection_SIMULATED.json` |
| — | M9 IEEE manuscript | ✅ Done (source; no PDF — B-003) | `paper/main.tex`, 8 paper-vs-artefact tests |
| — | M10 Reproducibility + final tests | ✅ Done | `scripts/run_all.py`, 41 tests green |

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

## Completion status

| Requirement | State | Evidence |
|---|---|---|
| CRISP-DM pipeline documented and implemented | ✅ | `research/crisp_dm/01…06`, `scripts/run_all.py` (17 stages) |
| XGBoost trained and evaluated on valid data | ✅ | `results/metrics/holdout_test_final.json`, real LEAD1.0 annotations |
| Backend and interactive dashboard work | ✅ | `backend/app/main.py`, 13 API tests, dashboard verified in a browser |
| Simulation module works | ✅ | `src/simulation/`, 4 equipment models, transfer test scored |
| IEEE manuscript contains verified results | ✅ (source) | `paper/main.tex` + 8 paper-vs-artefact tests; **no PDF** (B-003) |
| Tests and reproducibility checks pass | ✅ | 41 tests green; full pipeline re-run reproduced identical metrics |
| Documentation complete | ✅ | README, ARCHITECTURE, PLAN, PROGRESS, DECISIONS (13), EXPERIMENT_LOG, BLOCKERS |

## Remaining optional work (not blockers)

1. Compile `paper/main.tex` once a TeX distribution exists (B-003), then record page count and fix
   any float placement.
2. Repeat the evaluation over several building-disjoint splits to replace the single-split estimate
   with a distribution.
3. Add drift detection and a rolling threshold recalibration before any pilot deployment.

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
- 2026-11-09 — M6/M7 complete: FastAPI service (`/health`, `/model`, `/score`, `/batch`, `/simulate`,
  `/dashboard-data`) with a single scoring path shared by training and serving (`src/inference/service.py`),
  plus a dependency-free canvas dashboard served from `backend/app/static/index.html`. 13 API tests.
- 2026-11-09 — M8 complete: four geothermal auxiliary-equipment models (separator, feed pump, bearing,
  compressor) generate SIMULATED hourly signals with injected, labelled faults; the frozen model scores
  them through the same inference path. Result: 0.954 ROC-AUC on the separator signal but **0 of 252
  fault hours flagged** at the plant-calibrated threshold — a negative transfer result, reported as such.
- 2026-11-09 — M9 complete (source only): `paper/main.tex`, IEEE conference format, 10 sections,
  8 bibliography entries, 4 figures. `tests/test_paper_numbers.py` (8 tests) fails if any headline
  number drifts from `results/metrics/*.json`. No PDF: no LaTeX distribution here (B-003).
- 2026-11-09 — M10 complete: `scripts/run_all.py` runs 17 stages in dependency order and writes
  `results/metrics/pipeline_run.json`. A full re-run from `--from holdout` reproduced the held-out
  metrics exactly (ROC-AUC 0.9524, PR-AUC 0.8342, Brier 0.00613, threshold 0.1283, 441/536 events),
  confirming the training path is deterministic given the pinned seeds. 41 tests green.
- 2026-11-09 — Integrity fix: `results/metrics/optuna_xgb_best.json` renamed to `tuning_best.json`,
  because the artefact came from the deterministic grid, not the TPE study (D-013). All references
  updated; no stale paths remain (`grep -rn optuna_xgb_best` returns nothing).
- 2026-11-09 — Dashboard fix: the telemetry chart had no x-axis reference; five MM-DD ticks are now
  drawn from the timestamps the API returns (verified in a browser: 01-01, 02-20, 06-22, 08-19, 09-29).
- 2026-11-09 — `src/models/tune.py` fix: a locked Optuna store no longer kills the run. Two concurrent
  tuning processes collided on `experiments/optuna_xgb.db` and the second died with
  `PermissionError [WinError 32]`; it now falls back to a timestamped database.
- 2026-11-09 — Security note: `dahl_keys.txt` (plaintext API bearer tokens) and `check_dahl.sh` were
  found untracked in the project root. Both are now in `.gitignore`; they were never committed and were
  not executed. The tokens should be rotated if that directory was ever shared or pushed.
