# EXPERIMENT_LOG

Append-only. Every training/evaluation run gets an entry with: id, date, purpose, data split,
config, metrics, artifact paths, and whether it was a *selection* run (used to choose something)
or a *reporting* run (final, test set touched).

Metrics are copied from files under `results/metrics/` — never typed from memory.

---

## E-000 — Data audit (baseline of record)

- **Date:** 2026-11-09
- **Command:** `python -m src.data.audit`
- **Purpose:** Establish verified dataset facts before any modelling.
- **Result:** 1,749,494 rows / 200 buildings / 8,784 hours / 2.13% anomaly rate / 45.91:1 imbalance /
  6.15% missing readings / 5,503 anomaly events (median 2 h).
- **Artifacts:** `results/metrics/data_audit.json`, `results/tables/data_audit_summary.md`,
  `results/tables/per_building_audit.csv`
- **Type:** descriptive (no model)

---

## E-001 — Causal feature build + building-disjoint splits

- **Date:** 2026-11-09
- **Command:** `python -m src.features.build_features && python -m src.features.splits && python -m pytest tests/test_features_no_leakage.py -q`
- **Purpose:** Produce the modelling matrix and freeze the evaluation protocol before any model is trained.
- **Split / seed:** 140 / 30 / 30 buildings (D-010); 5 grouped folds inside train.
- **Config:** 36 features, 7 families; past-only windows 6/24/168/720 h with `min_periods` 3/12/24/48; `EPS = 1e-6`; per-building hourly-grid reindexing (D-009).
- **Metrics:** 1,749,494 rows × 36 features. Anomaly rate 2.162% / 2.049% / 2.072% (train/val/test). Median ACF 0.933 (lag 1 h), 0.681 (lag 24 h). Feature NaN rate 0%–9.7%. 6/6 leakage and split tests pass.
- **Artifacts:** `data/processed/features.parquet`, `data/processed/splits.json`, `data/processed/splits_summary.csv`, `results/metrics/feature_metadata.json`
- **Type:** descriptive (no model)

---

<!-- New entries below. Template:

## E-NNN — <title>

- **Date:**
- **Command:**
- **Purpose:**
- **Split / seed:**
- **Config:**
- **Metrics:** (copy from results/metrics/…)
- **Artifacts:**
- **Type:** selection | reporting | diagnostic

-->
