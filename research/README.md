# Research documentation — CRISP-DM

Six phase reports, written as the work was done rather than reconstructed afterwards. Every number in
them comes from a file under `results/` that a script in `src/` produces; none were typed from a
terminal by hand, and `tests/test_paper_numbers.py` enforces the same rule for the manuscript.

| Phase | Document | What it establishes |
|---|---|---|
| 1 Business understanding | [`01_business_understanding.md`](crisp_dm/01_business_understanding.md) | Why geothermal auxiliary equipment is the target, why labels are scarce, what the maintenance cost asymmetry is, and the success criteria. |
| 2 Data understanding | [`02_data_understanding.md`](crisp_dm/02_data_understanding.md) | 1.75 M rows, 200 buildings, 2.13% anomalies, 45.9:1 imbalance, 6.15% missing, 5,503 episodes; lag-1 autocorrelation 0.933 — the empirical case for causal features and building-disjoint evaluation. |
| 3 Data preparation | [`03_data_preparation.md`](crisp_dm/03_data_preparation.md) | The causal feature contract and its truncation test, 34 features in six families, the 140/30/30 split and why it is balanced by anomaly rate. |
| 4 Modeling | [`04_modeling.md`](crisp_dm/04_modeling.md) | Baselines (logistic 0.809, Isolation Forest 0.806), XGBoost 0.956 validation ROC-AUC, a flat hyper-parameter surface, the measured cost of class weighting, and the family ablation showing dynamics dominate. |
| 5 Evaluation | [`05_evaluation.md`](crisp_dm/05_evaluation.md) | The single test-set read: 0.9524 ROC-AUC (CI 0.9249–0.9702), 0.8342 PR-AUC, 82.3% episode recall at 1 h latency, ECE 0.0005, per-building recall 0.02–1.00, threats to validity. |
| 6 Deployment | [`06_deployment.md`](crisp_dm/06_deployment.md) | One scoring path shared by API and training, the dashboard, the **SIMULATED** geothermal module, and the transfer test that flags 0 of 252 fault hours. |

## The four decisions that shaped the study

1. **Causal features, proven by test** (D-008). Median lag-1 autocorrelation of 0.933 means a
   row-wise split would produce a meaningless result. Features are rebuilt on truncated history and
   must be identical at every time ≤ the cut; a second test perturbs only the current reading.
2. **Building-disjoint evaluation** (D-002, D-010). 140/30/30 buildings, balanced by anomaly rate, so
   the reported number answers "will this catch faults in an asset we have not instrumented?".
3. **Observed readings only** (D-011). No missing hour in LEAD1.0 is labelled anomalous, so keeping
   them teaches "silent = healthy". Outages became a separate alarm class.
4. **No cross-asset features** (D-012). A peer feature broke train/serve consistency by 57% and was
   removed; every feature must be computable from one asset's own history.

## Reading order for a reviewer

1. `02_data_understanding.md` §2.4–2.5 — what the labels actually look like (episodic, not i.i.d.).
2. `03_data_preparation.md` §3.1 — the contract, and the test that enforces it.
3. `04_modeling.md` §4.6 — which feature families carry the signal.
4. `05_evaluation.md` §5.4 — where the model fails, per building.
5. `06_deployment.md` §6.4 — the negative transfer result.

## Negative results, collected

Kept in one place because they are the most transferable part of the work:

- Calendar features: ROC-AUC 0.729 but PR-AUC 0.064 — they rank rows without identifying anomalies.
- Class weighting (`scale_pos_weight` = 42.4): +1.6 points episode recall, −0.4 ROC-AUC, and a Brier
  score 3.5× worse. Rejected.
- Hyper-parameter search: six candidates within 0.0026 PR-AUC. The surface is flat; tuning is not the
  lever.
- Cross-asset transfer: 0 of 252 injected fault hours flagged at the frozen threshold, despite a
  0.954 ROC-AUC on one signal. A probability threshold does not transfer across asset populations.
- Per-building recall: 4 of 30 held-out buildings below 0.5, worst 0.017. The aggregate AUC hides a
  bimodal distribution.
