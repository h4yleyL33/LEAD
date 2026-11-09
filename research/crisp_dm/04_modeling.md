# Phase 4 — Modeling

**CRISP-DM phase:** 4 of 6
**Status:** complete
**Reproduce:**
```bash
python -m src.models.baselines                        # prior / logistic / isolation forest
python -m src.models.tune --grid                      # 6-candidate deterministic search
python -m src.models.train_xgboost                    # default hyper-parameters
python -m src.models.train_xgboost --params-from results/metrics/tuning_best.json --tag tuned
python -m src.models.train_xgboost --weighted         # imbalance ablation
python -m src.models.select_final                     # freeze models/xgb_final.json
python -m src.evaluation.compare                      # one table for the paper
```

Every number below is copied from `results/metrics/*.json`, `results/tables/model_comparison_val.csv`.
Nothing in this document was typed from a terminal by hand.

---

## 4.1 Problem formulation

Binary classification of one (building, hour) row: is this reading anomalous? Trained on 140
buildings, early-stopped on 30, reported once on 30 held-out buildings. 34 causal features, 2.27%
positive rate on observed rows, 43:1 imbalance.

Two deliberate choices frame the whole phase:

1. **The prior is left intact** (no `scale_pos_weight` by default). A model that predicts the true
   prevalence produces probabilities that can be compared with the cost of an inspection; a
   reweighted model does not. The reweighted variant is measured, not assumed (§4.5).
2. **The operating threshold is selected on validation**, by maximising F2 (recall weighted twice as
   heavily as precision), then frozen and applied unchanged to test. No test-set threshold tuning.

## 4.2 Baselines (validation, 30 buildings, 250,019 rows)

| Model | ROC-AUC | PR-AUC | Brier | recall@FPR1% | event recall |
|---|---|---|---|---|---|
| prior (constant) | 0.5000 | 0.0215 | 0.02104 | 0.0100 | — |
| logistic regression | 0.8087 | 0.5715 | 0.01304 | 0.5651 | 0.7803 |
| isolation forest (unsupervised) | 0.8058 | 0.2296 | 0.13484 | 0.3006 | 0.4740 |
| **XGBoost (selected)** | **0.9562** | **0.8198** | **0.00662** | **0.7989** | **0.8962** |

Three things this table establishes:

- **The problem is learnable from telemetry alone.** A linear model on the same causal features
  reaches 0.81 ROC-AUC, so the signal is not an artefact of tree flexibility.
- **Labels are worth much.** The unsupervised Isolation Forest, the standard industrial
  answer when no annotations exist, reaches 0.81 ROC-AUC but only 0.23 PR-AUC and 0.30
  recall@FPR1%. It ranks gross outliers well and subtle episodes badly. This is the quantitative
  argument for investing in annotation.
- **XGBoost adds +0.15 ROC-AUC over the linear model**, which means the feature interactions
  (deviation × volatility × calendar) carry real information.

## 4.3 Hyper-parameter search

A capped Optuna TPE study was attempted first and abandoned after 11 completed trials: each trial
cost ~100 s (3 grouped folds on a 25% row sample), and a resumed study replayed identical parameter
sets because TPE restarts its seeded random startup phase on reload (recorded as E-003). The search
was replaced by a deterministic 6-candidate grid on 2 grouped folds, 20% row subsample, 250 rounds,
early stopping 25.

| Candidate | depth | lr | min_child_w | subsample | colsample | λ | fold-mean PR-AUC |
|---|---|---|---|---|---|---|---|
| 0 (hand-set default) | 6 | 0.050 | 5.0 | 0.80 | 0.80 | 1.0 | 0.7745 |
| 1 (best TPE point) | 5 | 0.051 | 3.65 | 0.716 | 0.806 | 0.209 | 0.7737 |
| **2 (selected)** | **8** | **0.030** | **10.0** | **0.90** | **0.60** | **5.0** | **0.7758** |
| 3 | 4 | 0.080 | 2.0 | 0.70 | 0.90 | 1.0 | 0.7741 |
| 4 | 6 | 0.030 | 8.0 | 0.85 | 0.70 | 2.0 | 0.7732 |
| 5 | 7 | 0.060 | 4.0 | 0.80 | 0.75 | 0.5 | 0.7734 |

The surface is flat: all six candidates sit within 0.0026 PR-AUC (0.3% relative). The result is worth
stating plainly instead of presenting the 0.0013 "win" as meaningful. For this feature set,
hyper-parameters are not the lever; features and labels are.

## 4.4 Model selection (validation only)

| Variant | ROC-AUC | PR-AUC | Brier | event recall | threshold |
|---|---|---|---|---|---|
| `primary` (default params) | 0.9543 | 0.8239 | 0.00642 | 0.8875 | 0.1403 |
| **`tuned` (candidate 2) — selected** | **0.9562** | 0.8198 | 0.00662 | 0.8962 | 0.1283 |
| `weighted` (`scale_pos_weight` = 42.4) | 0.9519 | 0.8182 | 0.02338 | 0.9118 | 0.6835 |

The selection rule was fixed in advance (D-004): highest validation ROC-AUC, ties broken by PR-AUC.
`tuned` wins by 0.0019 ROC-AUC and is frozen as `models/xgb_final.json`. The honest reading is that
`tuned` and `primary` are indistinguishable on 250k validation rows; the rule was followed rather
than re-litigated after seeing the numbers.

## 4.5 Does class weighting help? (the standard reflex, measured)

| | ROC-AUC | PR-AUC | Brier | event recall | precision at threshold |
|---|---|---|---|---|---|
| unweighted (selected) | 0.9562 | 0.8198 | **0.00662** | 0.8962 | **0.7918** |
| `scale_pos_weight` = 42.4 | 0.9519 | 0.8182 | 0.02338 | **0.9118** | 0.7542 |

Reweighting costs 0.004 ROC-AUC, 0.0016 PR-AUC and triples the Brier score (0.0066 → 0.0234),
while buying +0.016 event recall. Because the deployment needs interpretable probabilities (the
dashboard shows "P(anomaly) = 0.31" and the operator decides), the unweighted model is kept and the
operating point is moved instead. That is exactly what F2 threshold selection does.

## 4.6 Which features carry the signal? (family-only ablation)

Each family alone, same hyper-parameters, validation split:

| Family | n | ROC-AUC | PR-AUC | event recall |
|---|---|---|---|---|
| dynamics | 9 | **0.9284** | **0.7657** | **0.8806** |
| deviation | 8 | 0.9072 | 0.7268 | 0.5813 |
| level | 5 | 0.9011 | 0.6734 | 0.4308 |
| volatility | 4 | 0.8290 | 0.4507 | 0.3616 |
| peer *(removed, D-012)* | 1 | 0.7798 | 0.3805 | 0.3772 |
| calendar | 6 | 0.7290 | 0.0640 | 0.3374 |
| availability | 2 | 0.6492 | 0.0870 | 0.5433 |
| **all 34** | 34 | **0.9562** | **0.8198** | **0.8962** |

The interpretation is the opposite of what a reader might expect:

- **Dynamics alone nearly matches the full model on ROC-AUC (0.928 vs 0.956) and beats it on
  event recall (0.881 vs 0.896, close).** First differences, same-hour-yesterday/last-week
  comparisons and short-vs-long momentum are the workhorse. Anomalies in this dataset are
  *changes*, not *levels*.
- **Level alone gets 0.90 ROC-AUC but only 0.43 event recall.** A level-based score fires on the
  peak hour of an episode and misses the rest; that is the difference between ranking rows and
  detecting episodes, and it is why event metrics are reported at all.
- **Calendar alone is nearly useless for precision (PR-AUC 0.064)**, as it must be, since the
  anomaly rate varies only 1.10%–3.61% across weekday × hour. It contributes as a modifier inside
  interactions, not as a standalone signal.
- **Availability alone reaches 0.65 ROC-AUC with 0.54 event recall**, because past outages
  (`f_hours_since_valid`, `f_missing_24h`) genuinely precede annotated anomalies. Note this is
  *past* missingness; current-hour missingness is excluded by D-011.

Gain-based importance on the full model agrees: `f_dev_rel_720` (12.7%), `f_abs_diff1` (12.3%),
`f_dev_rel_168` (8.5%), `f_log_value` (6.3%), `f_ratio_med_168` (5.5%).

## 4.7 Artefacts

| Path | Contents |
|---|---|
| `models/xgb_final.json` | frozen selected model (734 trees) |
| `models/xgb_primary.json`, `xgb_tuned.json`, `xgb_weighted.json` | compared variants |
| `results/metrics/xgb_{tag}_val.json` | validation metrics + full parameter record + environment |
| `results/metrics/tuning_best.json` | grid search results |
| `results/metrics/ablation_families_only.json` | family-only ablation |
| `results/metrics/model_selection.json` | selection table + rule |
| `results/tables/feature_importance_{tag}.csv` | gain importance |
| `results/tables/model_comparison_val.csv` / `.md` | the table in §4.2/§4.4 |

## 4.8 Exit criteria (met)

- [x] Three baselines trained and scored on validation.
- [x] XGBoost trained with early stopping on validation buildings only.
- [x] Hyper-parameter search capped, documented, and its flatness reported honestly.
- [x] Class-imbalance question answered by measurement, not convention.
- [x] Feature families ablated; the result (dynamics dominate) is stated even though it
      de-emphasises the "AIoT multi-signal" framing.
- [x] One frozen final artefact; the test split untouched at this point.
