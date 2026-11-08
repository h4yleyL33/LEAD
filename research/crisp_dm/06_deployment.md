# Phase 6 — Deployment

**CRISP-DM phase:** 6 of 6
**Status:** backend, dashboard and simulation module complete and tested
**Reproduce:**
```bash
python -m src.simulation.geothermal          # -> data/synthetic/*_SIMULATED.csv
python -m src.simulation.score_simulated     # -> results/metrics/simulation_detection_SIMULATED.json
python -m uvicorn backend.app.main:app --port 8000
# open http://127.0.0.1:8000/
python -m pytest backend/tests -q            # 13 tests
```

---

## 6.1 Architecture

```
data/processed/features.parquet ─┐
models/xgb_final.json            ├─► src/inference/service.py ─► backend/app/main.py ─► dashboard
results/metrics/*.json           ┘        (single scoring path)      (FastAPI)          (canvas UI)
                                                  ▲
src/simulation/geothermal.py ─────────────────────┘   SIMULATED signals only
```

The single most important design decision: **there is exactly one scoring path.** The API, the
dashboard, the batch endpoint and the simulation all call `AnomalyScorer`, which calls the same
`build_feature_frame` used to build the training matrix. `tests/test_inference.py` asserts that
serving and batch agree to 1e-6 on a real row, so the two cannot drift apart silently.

## 6.2 API surface

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness, loaded model tag, threshold, feature count |
| GET | `/model` | model card: metrics, features, selection rule, caveats |
| GET | `/metrics` | held-out test metrics (the numbers in the paper) |
| GET | `/buildings` | the 30 held-out buildings with per-building recall |
| GET | `/buildings/{id}/series` | readings + scores + labels, downsampled to ≤ 9000 points |
| POST | `/score` | score the last hour of a caller-supplied history (≥ 24 readings) |
| POST | `/batch` | score many buildings from one long frame |
| GET | `/simulate` | **SIMULATED** geothermal series with injected-fault overlay |
| GET | `/` | interactive dashboard |

`POST /score` is what makes the system usable on a plant that is not in the training set: the caller
supplies history, the server rebuilds the features with the training-time code, and returns a
probability, an alarm class and the threshold used.

**Alarm classes.** `ANOMALY` (score ≥ threshold), `NORMAL`, and `OUTAGE` — an hour with no reading.
Outages are never scored (D-011); a dead sensor is a fault, but it is a *different* fault from a
consumption anomaly, and merging the two would hide it.

## 6.3 Dashboard

Single static HTML page, no build step, no CDN (works offline). Three panels:

1. **KPI strip** — ROC-AUC with CI, PR-AUC vs base rate, event recall, median latency, Brier/ECE,
   false alarms per building-day. Read from `/metrics`, so the page cannot drift from the paper.
2. **Held-out building explorer** — reading series, annotated anomalies, flagged hours as shaded
   bands, and a score strip with the threshold line. A slider moves the threshold and recomputes
   precision/recall *for that building* as a what-if.
3. **SIMULATED geothermal panel** — the four synthetic auxiliary-equipment signals with the injected
   fault window shaded, scored by the same model. Header text says SIMULATED; the payload carries a
   `warning` field; the page's data notice repeats it.

The building selector defaults to the *median-performing* held-out building and labels each option
with its recall, because per-building recall ranges 0.02–1.00 and opening on a lucky building would
misrepresent the system.

## 6.4 Geothermal simulation (SIMULATED)

Four auxiliary signals over 90 days (2,160 h), hourly, with four injected faults of known extent:

| Signal | Unit | Fault injected | Extent |
|---|---|---|---|
| cooling-water pump power | kW | cavitation (oscillation + turbulence) | 36 h |
| NCG compressor current | A | fouling (gradual rise) | 120 h |
| turbine bearing temperature | °C | overheat ramp + trip spike | 72 h |
| separator level | % | controller drift (step) | 24 h |

Generators use plausible parameter values, not fitted ones. Every output row carries
`simulated = True`; filenames carry `_SIMULATED`; the figure title states it; the API payload states
it. **No result from this module may be reported as a PLTP Kamojang measurement (D-005).**

### What happens when the building-trained model meets these signals

| Equipment | ROC-AUC | PR-AUC | fault hours flagged at the frozen threshold |
|---|---|---|---|
| separator (level drift) | 0.954 | 0.110 | 0 / 24 |
| pump (cavitation) | 0.868 | 0.142 | 0 / 36 |
| bearing (overheat) | 0.579 | 0.094 | 0 / 72 |
| compressor (fouling) | 0.389 | 0.042 | 0 / 120 |

Aggregate: 0 of 252 fault hours flagged; mean ROC-AUC 0.697, mean PR-AUC 0.097.

This is the honest headline of the simulation, and it is not flattering:

- **The operating threshold does not transfer.** Scores on the simulated signals sit an order of
  magnitude below the building-calibrated 0.1283, so the alarm never fires even where ranking is
  good (separator: ROC-AUC 0.95, zero alarms). A frozen absolute threshold is only valid for the
  population it was calibrated on.
- **Ranking transfers only partially** — well for a slow drift, badly for compressor fouling, whose
  ramp is small relative to that signal's own noise.
- **Conclusion:** deploying on geothermal equipment requires per-asset recalibration and, more
  likely, retraining on plant data. The simulation demonstrates that the pipeline runs end to end on
  a different asset class; it does not demonstrate geothermal performance, and it was deliberately
  not used to tune anything.

## 6.5 Operational behaviour

| Property | Value | Source |
|---|---|---|
| Scoring latency, single building history | < 1 s (2,160–8,784 rows, feature build included) | measured during API tests |
| Model size | 734 trees, ~1.4 MB JSON | `models/xgb_final.json` |
| Cold start | ~6 s (parquet index + model load) | startup log |
| Test suite | 33 tests (20 pipeline + 13 API) | `pytest backend/tests tests -q` |

Monitoring hooks that exist: `/health` for liveness, `/metrics` for the frozen evaluation numbers,
and per-building recall in `/buildings` so a degradation in one asset is visible rather than averaged
away. There is no drift detector — stated as a limitation, not pretended away.

## 6.6 What is deliberately absent

- **No authentication, no TLS, no persistence.** This is a research artefact bound to
  `127.0.0.1`. Exposing it would require auth, rate limiting and a database; none of that is claimed.
- **No live SCADA connector.** There is no plant to connect to (B-002), and inventing one would
  produce a demo that lies about its data source.
- **No retraining loop.** The model is frozen by design so the reported test numbers stay true.

## 6.7 Exit criteria (met)

- [x] Backend serves model card, metrics, per-building series, single and batch scoring.
- [x] Dashboard renders real held-out data and states its provenance on the page.
- [x] Simulation module generates documented synthetic signals with ground truth, clearly labelled.
- [x] Simulation results reported as they came out, including the failure to transfer.
- [x] 13 API tests + 20 pipeline tests green.
