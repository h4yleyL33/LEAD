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

## E-002 — Reference baselines on the validation buildings

- **Date:** 2026-11-09
- **Command:** `python -m src.models.baselines`
- **Purpose:** Establish floor (prior), linear reference (logistic) and unsupervised reference (Isolation Forest).
- **Split / seed:** val (30 buildings, 250,019 observed rows); seed 42.
- **Config:** median imputation + missing indicators + standardisation for logistic (lbfgs, 300 iters); IForest 200 trees, max_samples 4096, contamination auto.
- **Metrics:** prior ROC-AUC 0.5000 / Brier 0.02104; logistic 0.8087 / PR-AUC 0.5715 / event recall 0.7803; iforest 0.8058 / PR-AUC 0.2296 / recall@FPR1% 0.3006.
- **Artifacts:** `results/metrics/baselines_val.json`
- **Type:** reporting

## E-003 — Hyper-parameter search (TPE abandoned, deterministic grid used)

- **Date:** 2026-11-09
- **Command:** `python -m src.models.tune --grid`
- **Purpose:** Tune XGBoost on grouped folds inside the training buildings.
- **Split / seed:** 2 grouped folds (28 buildings each), 20% row subsample; seed 42.
- **Config:** 6 explicit candidates, 250 rounds, early stopping 25, metric = fold-mean PR-AUC.
- **Metrics:** candidate PR-AUC 0.7732–0.7758 (spread 0.3% relative). Selected candidate 2 (depth 8, lr 0.03, λ 5).
- **Artifacts:** `results/metrics/tuning_best.json`, `experiments/optuna_xgb_DISCARDED_replay.db`
- **Type:** selection
- **Notes:** the TPE study was stopped after 11 trials (~100 s each) and is **invalid**: resuming a persisted study restarts TPE's seeded random startup phase, so trials 5–7 exactly replayed the parameter sets of trials 0–2. The DB is kept as evidence, not used. Fix recorded in `src/models/tune.py` (`--fresh` default, explicit warning on resume).

## E-004 — Feature-family ablation

- **Date:** 2026-11-09
- **Command:** `python -m src.models.ablation --families level,deviation,dynamics,volatility` then `--families availability,calendar,peer`
- **Purpose:** Attribute the signal to physical feature families.
- **Split / seed:** val; seed 42.
- **Config:** one model per family, 800 rounds, early stopping 50.
- **Metrics:** dynamics 0.9284 / 0.7657; deviation 0.9072 / 0.7268; level 0.9011 / 0.6734; volatility 0.8290 / 0.4507; peer 0.7798 / 0.3805; calendar 0.7290 / 0.0640; availability 0.6492 / 0.0870 (ROC-AUC / PR-AUC).
- **Artifacts:** `results/metrics/ablation_families_only.json`, `results/tables/ablation_families_only.csv`
- **Type:** diagnostic
- **Notes:** the `peer` row was measured with `include_peer=True` (v1 feature set); the feature was later removed (D-012) and is not part of the final model.

## E-005 — Final model selection and single test-set evaluation

- **Date:** 2026-11-09
- **Command:** `python -m src.models.train_xgboost` → `--params-from ... --tag tuned` → `--weighted` → `python -m src.models.select_final` → `python -m src.evaluation.holdout --variant final`
- **Purpose:** Choose the final model on validation, then report the held-out result once.
- **Split / seed:** train 140 / val 30 / test 30 buildings; seed 42.
- **Config:** XGBoost hist, early stopping 75 on validation aucpr; final = candidate 2 params (depth 8, lr 0.03, min_child_weight 10, subsample 0.9, colsample 0.6, λ 5, α 0.1), 734 trees.
- **Metrics (test, 30 buildings, 243,684 rows):** ROC-AUC 0.9524 (95% CI 0.9249–0.9702), PR-AUC 0.8342, Brier 0.00613, ECE 0.0005, recall@FPR1% 0.8089, precision/recall 0.7987/0.7888 at threshold 0.1283, event recall 0.8228 (441/536 events), median latency 1 h, 0.056 false-alarm episodes per building-day, median per-building recall 0.842.
- **Artifacts:** `models/xgb_final.json`, `results/metrics/holdout_test_final.json`, `results/predictions/test_predictions_final.parquet`, `results/tables/holdout_per_building_final.csv`, 4 figures, `results/tables/model_comparison_{val,test}.csv`
- **Type:** reporting

---

<!-- Template for new entries:

## E-006 — End-to-end reproducibility re-run

- **Date:** 2026-11-09
- **Command:** `python scripts/run_all.py --from holdout --keep-going`
- **Purpose:** Verify that the pipeline regenerates the reported numbers from the committed code, not just that the artefacts exist.
- **Split / seed:** unchanged (train 140 / val 30 / test 30 buildings; seed 42).
- **Config:** identical to E-005; the models were retrained from scratch earlier in the same session (`--fast` skips only the tuning grid and the weighted ablation).
- **Metrics (test):** ROC-AUC 0.9524, PR-AUC 0.8342, Brier 0.00613, ECE 0.0005, recall@FPR1% 0.8089, precision/recall 0.7987/0.7888 at threshold 0.1283, event recall 0.8228 (441/536), 0.056 false alarms per building-day — **identical to E-005 at the reported precision**.
- **Stage timings:** holdout 35.4 s, compare 0.8 s, SHAP 26.1 s, simulation 0.8 s, transfer test 7.2 s, paper validation 0.2 s, tests 9.5 s.
- **Artifacts:** `results/metrics/pipeline_run.json`, all of `results/` regenerated, `paper/figures/` refreshed.
- **Type:** diagnostic

---

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
