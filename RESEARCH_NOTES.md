
## 2026-11-09 — Full pipeline verification

`scripts/run_all.py` executed all stages end-to-end in dependency order
(audit → EDA → features → splits → leakage → baselines → train → tuned →
select → holdout → compare → SHAP → simulate → transfer test → paper → tests).
See `EXPERIMENT_LOG.md` E-007 and `results/metrics/pipeline_run.json` for the
machine-readable record.
