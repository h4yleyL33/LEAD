# Research Notes: Energy Consumption Anomaly Detection, AIoT Project

## Dataset

- **LEAD1.0-small** (D-001): 200 buildings, year 2016, ~1.75 M rows, 8,784 h hourly grid
  (`results/metrics/data_audit.json`)
- **Missing readings:** 6.15% (D-009: per-building grid reindex; D-011: observed-only
  scoring, no imputation for the tree model)
- **Target:** binary anomaly flag, base rate ~2.1% (1:46 class imbalance)

## Key Decisions

| # | Decision |
|---|---|
| D-001 | LEAD1.0-small (200 buildings) as primary real dataset |
| D-002 | Building-disjoint split 140/30/30 (train/val/test) |
| D-003 | 35 causal (backward-looking) features, 6 families |
| D-004 | ROC-AUC primary; PR-AUC + recall@FPR secondary |
| D-005 | Geothermal data SIMULATED and labelled as such |
| D-006 | Python 3.14 + xgboost 3.4.1 (no downgrade) |
| D-007 | No imputation for tree model; median for linear baselines |
| D-008 | Causal contract enforced by executable truncation test |
| D-009 | Grid reindexing per building before rolling windows |
| D-010 | GroupShuffleSplit for temporal dependency |
| D-011 | Observed readings only; outages = separate alarm class |
| D-012 | No cross-building "peer" features |
| D-013 | Deterministic 6-candidate grid over Optuna TPE |

## Limitations

1. **No PLTP Kamojang data**: the geothermal transfer test uses synthetic signals only (D-005).
   Real deployment needs on-site retraining and recalibration.
2. **No Kaggle leaderboard access** (B-001): no external score comparison.
3. **One building-disjoint split**: uncertainty quantified only via 300-sample bootstrap
   on the test split (CI95 0.9249-0.9702 ROC-AUC).
4. **Threshold tuned on validation**: transfer to a new population is not guaranteed
   (confirmed empirically by the simulated transfer test).

## Negative Results

| Experiment | Result | Interpretation |
|---|---|---|
| Calendar features only | PR-AUC 0.0640 | Time-of-day/day-of-week carries no anomaly signal alone |
| Class weighting | Brier 0.02338 vs 0.00642 (3.6x worse) | Lifts event recall (0.912) but destroys calibration — rejected |
| Optuna TPE (hundreds of trials) | Spread 0.0026 PR-AUC | Flat surface; deterministic grid sufficient (D-013) |
| Cross-asset threshold transfer | 1/252 fault hours flagged at frozen threshold | Operating threshold does not generalize across asset classes |

## Feature Family Ablation (validation, families-only models)

| Family | n_feat | ROC-AUC | PR-AUC | event_recall |
|---|---|---|---|---|
| dynamics | 9 | 0.9284 | 0.7657 | 0.8806 |
| deviation | 8 | 0.9072 | 0.7268 | 0.5813 |
| level | 5 | 0.9011 | 0.6734 | 0.4308 |
| volatility | 4 | 0.8290 | 0.4507 | 0.3616 |
| peer | 1 | 0.7798 | 0.3805 | 0.3772 |
| availability | 2 | 0.6492 | 0.0870 | 0.5433 |
| calendar | 6 | 0.7290 | 0.0640 | 0.3374 |

Dynamics (rolling z-scores, momentum, CV) carries most of the signal alone. Final model
excludes availability (D-011) and peer (D-012) families.

## Final Model Performance

Validation (30 buildings, 250,019 rows; `results/metrics/xgb_primary_val.json`):
ROC-AUC 0.9543, PR-AUC 0.8239, Brier 0.00642, event recall 0.888 (513/578),
median detection latency 1 h, false alarms 0.048/building-day.

Test (30 held-out buildings, 243,684 rows; `results/metrics/holdout_test_final.json`):
ROC-AUC 0.9524 (95% CI 0.9249-0.9702, 300 bootstrap), PR-AUC 0.8342, Brier 0.00613,
event recall 0.823 (441/536), median latency 1 h, false alarms 0.056/building-day,
ECE-uniform-15bin 0.00093 (well calibrated).

Model selection, per `model_selection.json`: the rule "max validation ROC-AUC, tie-break
PR-AUC" selected the tuned variant (ROC-AUC 0.9562, PR-AUC 0.8199, Brier 0.00662,
734 iterations) over primary (0.9543/0.8239) and weighted (0.9519/0.8182, Brier 0.02338).
Weighted lifted event recall to 0.9118 but destroyed calibration (Brier 3.6x worse), and it was rejected.

## Top SHAP Features (global, TreeExplainer, n=10,000)

`f_log_value` (0.567), `f_abs_diff1` (0.554), `f_cv_24` (0.287), `f_ratio_168` (0.252),
`f_range_24` (0.230); log-odds units; level and first-difference features dominate.

## Simulated Geothermal Transfer (smoke test only)

`simulation_detection_SIMULATED.json`: frozen validation threshold flags 1/252
simulated fault hours (bearing 0/72, compressor 0/120, pump 0/36, separator 1/24);
threshold-free ranking quality mean ROC-AUC 0.697, mean PR-AUC 0.097. Scores sit an
order of magnitude below the building-calibrated threshold, so threshold recalibration
is mandatory for any new asset class. Synthetic signals, not plant measurements (D-005).

## Paper Status

- Source: `paper/main.tex` (595 lines), complete
- PDF: not compiled (B-003: no LaTeX on this machine)
- Paper numbers verified against `results/metrics/*.json` by `tests/test_paper_numbers.py`

## Pipeline Run

`scripts/run_all.py` executed all stages end-to-end in dependency order
(audit -> EDA -> features -> splits -> leakage -> baselines -> train -> tuned ->
select -> holdout -> compare -> SHAP -> simulate -> transfer test -> paper -> tests).
See `EXPERIMENT_LOG.md` E-007 and `results/metrics/pipeline_run.json`.
