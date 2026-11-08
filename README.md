# AIoT Energy Anomaly Detection — Geothermal Auxiliary Equipment

Reproducible CRISP-DM research and software pipeline for anomaly detection on hourly energy
telemetry, with a transfer demonstration to geothermal (PLTP) auxiliary equipment.

> **Integrity note.** The machine-learning model is trained and evaluated on **real annotated
> data** (LEAD1.0-small). The geothermal equipment signals are **simulated** and are never
> presented as measurements from PLTP Kamojang. See `data/README.md` and `BLOCKERS.md`.

## What is here

| Path | Contents |
|---|---|
| `MASTER_PROMPT.md` | Authoritative specification |
| `PROJECT_PLAN.md` | Milestones across all six CRISP-DM phases, evaluation protocol, risk register |
| `PROGRESS.md` | Living status + verified dataset facts |
| `DECISIONS.md` | Architecture/research decision log (ADR-lite) |
| `EXPERIMENT_LOG.md` | Every run: config, seed, metrics, artifacts |
| `BLOCKERS.md` | External dependencies needing user action |
| `research/crisp_dm/` | Phase-by-phase documentation |
| `src/` | Data audit, EDA, features, models, evaluation, explainability, simulation |
| `backend/` | FastAPI service + contract tests |
| `frontend/` | Interactive dashboard |
| `results/` | Metrics (JSON), tables (CSV/MD), figures (PNG), predictions |
| `paper/` | IEEE manuscript |
| `tests/` | Leakage, reproducibility, unit tests |

## Quick start

```bash
# 1. Environment (Python 3.12+; verified on 3.14.9)
python -m venv .venv
.venv/Scripts/activate          # Windows
pip install -r requirements.txt

# 2. Data (real, annotated; ~58 MB, not committed)
python scripts/fetch_data.py

# 3. Audit + EDA
python -m src.data.audit
python -m src.data.eda

# 4. Features + splits (leakage tests run first)
python -m src.features.build_features
pytest tests/test_features_no_leakage.py -q

# 5. Train + evaluate
python scripts/train.py --config configs/xgb_main.yaml

# 6. Serve
uvicorn backend.app.main:app --reload --port 8000
# dashboard: open frontend/index.html (or the served route)
```

Full reproduction in one command:

```bash
python scripts/reproduce_all.py
```

## Dataset

**LEAD1.0-small** — 1,749,494 hourly readings, 200 commercial buildings, 2016, with human
anomaly annotations (2.13% positive rate). Published by Gulati & Arjunan (BuildSys 2022),
derived from the ASHRAE Great Energy Predictor III data. Downloaded from
`github.com/samy101/lead-dataset`; **not redistributed here**.

## Headline results

See `results/metrics/holdout_metrics.json` and `paper/`. Numbers in the manuscript are generated
by `scripts/build_paper_artifacts.py` from the metrics files — none are typed by hand.

## License / use

Research and educational use. Third-party dataset terms apply to the data.
