# AIoT Energy Anomaly Detection: Hourly Telemetry with a Geothermal Transfer Test

Reproducible CRISP-DM research and software pipeline for anomaly detection on hourly energy telemetry:
causal feature engineering, XGBoost, building-disjoint evaluation, a FastAPI service, an interactive
dashboard, a simulated geothermal auxiliary-equipment module, and an IEEE manuscript whose numbers are
verified against the artefacts by test.

> **Integrity note.** The model is trained and evaluated on real annotated data (LEAD1.0-small,
> 1.75 M hourly readings from 200 commercial buildings). The geothermal equipment signals are
> **SIMULATED** and are never presented as measurements from PLTP Kamojang. Every file, figure and
> API payload that contains them says so. See `data/README.md` and `BLOCKERS.md`.

## Headline results

Held-out test set: 30 buildings unseen by the model, 243,684 observed hours, 5,454 anomalies,
scored once with a frozen model and a validation-selected threshold:

| Metric | Value |
|---|---|
| ROC-AUC | **0.9524** (95% CI 0.9249–0.9702, building-level bootstrap) |
| PR-AUC | **0.8342** (base rate 2.24%) |
| Brier / ECE | 0.00613 / 0.0005 |
| Recall @ FPR 1% | 0.8089 |
| Precision / recall @ τ = 0.1283 | 0.7987 / 0.7888 |
| Episode recall | 0.8228 (441 of 536), median latency 1 h |
| False-alarm episodes | 0.056 per building-day |

Negative results are reported in full. Calendar features rank rows without detecting anomalies
(PR-AUC 0.064); class weighting triples the Brier score for 1.6 points of episode recall; per-building
recall spans 0.02–1.00; and a threshold calibrated on buildings flags 0 of 252 injected fault hours
in the simulated geothermal transfer test.

## Quick start

```bash
# 1. Environment (Python 3.12+; verified on CPython 3.14.9, Windows 11)
python -m venv .venv
.venv/Scripts/activate                 # Windows;  source .venv/bin/activate on POSIX
pip install -r requirements.txt

# 2. Data — real, annotated, ~58 MB, not committed (pinned by SHA-256)
python scripts/fetch_data.py

# 3. Everything: audit -> EDA -> features -> splits -> baselines -> training ->
#    selection -> holdout -> SHAP -> simulation -> paper validation -> tests
python scripts/run_all.py              # ~25 min on 12 cores
python scripts/run_all.py --fast       # skips the tuning grid and the weighted ablation

# 4. Serve
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
#   dashboard: http://127.0.0.1:8000/      API docs: http://127.0.0.1:8000/docs
```

Individual stages are all runnable on their own: `python -m src.data.audit`,
`python -m src.features.build_features`, `python -m src.models.train_xgboost`, and so on. The full
order and the dependencies between stages are listed in `scripts/run_all.py`.

## Repository layout

| Path | Contents |
|---|---|
| `MASTER_PROMPT.md` | Authoritative project specification |
| `PROJECT_PLAN.md` | Six CRISP-DM phases, milestones M0–M10, risk register |
| `PROGRESS.md` | Living status, verified dataset and model facts |
| `DECISIONS.md` | Decision log D-001…D-012, each with context and consequences |
| `EXPERIMENT_LOG.md` | Every run: command, split, seed, config, metrics, artefacts |
| `BLOCKERS.md` | B-001 Kaggle credentials, B-002 no plant data, B-003 no LaTeX |
| `research/crisp_dm/` | `01_business_understanding` … `06_deployment` |
| `src/data/` | `audit.py`, `eda.py`, `load.py` |
| `src/features/` | `build_features.py` (causal contract), `splits.py` (building-disjoint) |
| `src/models/` | `baselines.py`, `tune.py`, `train_xgboost.py`, `ablation.py`, `select_final.py` |
| `src/evaluation/` | `metrics.py`, `holdout.py`, `compare.py` |
| `src/explainability/` | `shap_explain.py` |
| `src/inference/` | `service.py` — the single scoring path used by the API |
| `src/simulation/` | `geothermal.py` (SIMULATED), `score_simulated.py` |
| `backend/app/` | FastAPI service + dependency-free dashboard (`static/index.html`) |
| `results/` | `metrics/` (JSON), `tables/` (CSV/MD), `figures/` (PNG), `predictions/` (parquet) |
| `paper/` | `main.tex` (IEEE conference), figure copies, build instructions |
| `tests/`, `backend/tests/` | 41 tests: leakage, metrics, inference, API, paper-vs-artefact |

## What makes this reproducible

- **Data pinned**: URL + SHA-256 for both archive and CSV, re-verified on download.
- **Dependencies pinned**: exact versions in `requirements.txt` (numpy 2.5.4, pandas 3.1.1,
  scikit-learn 1.9.1, xgboost 3.4.1, shap 0.54.1, fastapi 0.162.2).
- **Leakage is a test, not a promise**: `tests/test_features_no_leakage.py` rebuilds features on
  truncated history and requires bit-identical values at every time ≤ the cut.
- **Serving cannot drift from training**: `tests/test_inference.py` scores the same row through the
  API path and the stored feature matrix and requires agreement to 1e-6.
- **The paper cannot drift from the artefacts**: `tests/test_paper_numbers.py` fails if any headline
  number in `paper/main.tex` differs from `results/metrics/*.json`.
- **One test split read**: the model and its threshold were frozen before the test buildings were
  scored, and the selection rule (max validation ROC-AUC) was written down first.

## Known limitations

1. **No plant data.** B-002: the geothermal module is synthetic. The transfer test is evidence about
   the pipeline, and it is not evidence about Kamojang.
2. **No leaderboard comparability.** B-001: the Kaggle competition split needs credentials; the public
   annotated subset is used instead, so no published score is quoted next to ours.
3. **No compiled PDF.** B-003: no LaTeX in this environment. `paper/main.tex` is complete and
   structurally validated but has never been compiled; build it with `scripts/build_paper.py` or
   Overleaf.
4. **Not production software.** No authentication, no persistence, no drift detection, no live SCADA
   connector. The service binds to localhost for demonstration.

## License / use

Research and educational use. Third-party dataset terms apply to the data (`data/README.md`).
