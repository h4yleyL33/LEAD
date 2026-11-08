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
