# PROGRESS — Living Status

**Last updated:** 2026-11-09
**Current milestone:** M2 → M3 transition (Data Understanding complete, Data Preparation starting)

---

## Snapshot

| Phase | Milestone | Status | Evidence |
|---|---|---|---|
| — | M0 Foundation (repo, venv, docs, git) | ✅ Done | `git log`, `.venv`, this file |
| 1 Business Understanding | M1 Framing, KPIs, cost model | ✅ Done | `research/crisp_dm/01_business_understanding.md` |
| 2 Data Understanding | M2 Acquire + audit + EDA | ✅ Audit done; EDA figures pending | `results/metrics/data_audit.json` |
| 3 Data Preparation | M3 Features + splits + leakage tests | 🔄 In progress | — |
| 4 Modeling | M4 Baselines + XGBoost + tuning | ⬜ Pending | — |
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

## Blockers

- **B-001** Kaggle competition split unavailable (no API credentials). Non-blocking for science;
  see `BLOCKERS.md`.

## Next actions (ordered)

1. `src/data/eda.py` → EDA figures + `research/crisp_dm/02_data_understanding.md`.
2. `src/features/build_features.py` (causal windows) + `src/features/splits.py` (building-disjoint).
3. `tests/test_features_no_leakage.py` — must be green before any model is trained.
4. Baselines + XGBoost, grouped CV, Optuna (capped), log to `EXPERIMENT_LOG.md`.
5. Holdout evaluation + SHAP + threshold analysis.
6. FastAPI backend → dashboard → geothermal simulation.
7. IEEE manuscript from regenerated artifacts.

## Changelog

- 2026-11-09 — Repo initialised; venv (Python 3.14.9) with numpy 2.5.4, pandas 3.1.1,
  scikit-learn 1.9.1, xgboost 3.4.1, shap 0.54.1, optuna, fastapi. LEAD1.0-small downloaded and
  audited. Planning docs created.
