"""FastAPI application: anomaly scoring and monitoring API for the AIoT pipeline.

Endpoints
---------
GET  /health                     liveness + loaded model identity
GET  /model                      model card: metrics, features, threshold, selection record
GET  /metrics                    held-out test metrics as reported in the paper
GET  /buildings                  buildings available in the held-out split
GET  /buildings/{id}/series      hourly readings + scores + labels for one building
POST /score                      score the last hour of a supplied history
POST /batch                      score many buildings from one long frame
GET  /                           interactive dashboard (static, no build step)

Design notes
------------
* The scorer is loaded once at startup from `models/xgb_final.json`; the threshold comes from the
  validation metrics file, never from the request.
* `/score` accepts the caller's own history and runs the *same* feature builder used in training, so
  an external building can be scored without any retraining.
* Outages are reported as `OUTAGE`, not scored (D-011).
"""
from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load import load_features, load_splits, observed_only, split_frame  # noqa: E402
from src.inference.service import AnomalyScorer  # noqa: E402
from src.utils.config import (  # noqa: E402
    GROUP_COL, METRICS_DIR, SYNTHETIC_DIR, TARGET_COL, TIME_COL, VALUE_COL,
)

from backend.app import datasets as ds_store  # noqa: E402
from backend.app.live import get_hub
from backend.app.external_api import ExternalSource, SourceConfig  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"


class Reading(BaseModel):
    timestamp: str = Field(..., description="ISO-8601 hourly timestamp")
    meter_reading: float | None = Field(None, description="kWh; null means no reading")


class ScoreRequest(BaseModel):
    building_id: int = 0
    history: list[Reading] = Field(..., min_length=1)


class BatchRequest(BaseModel):
    records: list[dict] = Field(..., min_length=1,
                                description="rows with building_id, timestamp, meter_reading")


class LiveReading(BaseModel):
    building_id: int
    timestamp: str
    meter_reading: float | None = None


class LiveIngestRequest(BaseModel):
    records: list[LiveReading] = Field(..., min_length=1)


class DatasetCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    csv: str = Field(..., description="raw CSV text")


class DatasetScoreRequest(BaseModel):
    building_id: int | None = Field(
        None, description="score only this building; omit to score every building in the file"
    )


class DatasetRename(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)


STATE: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE["scorer"] = AnomalyScorer("final")
    STATE["hub"] = get_hub(STATE["scorer"])
    STATE["test"] = None  # loaded lazily; 240k rows
    STATE["source"] = ExternalSource(STATE["hub"], PROJECT_ROOT / "data/integrations/source.json")
    try:
        yield
    finally:
        STATE["source"].stop()
        STATE.clear()


app = FastAPI(
    title="AIoT Energy Anomaly Detection API",
    description=(
        "XGBoost anomaly detector for hourly building energy telemetry, trained on LEAD1.0-small "
        "(real ASHRAE GEP III readings with human anomaly annotations). Held-out ROC-AUC 0.9524. "
        "Geothermal equipment signals are SIMULATED and are never presented as plant measurements."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    if request.url.path.startswith("/connection"):
        # Validation errors must not echo an auth token or the submitted config.
        return JSONResponse(status_code=422, content={"detail": "Konfigurasi koneksi tidak valid. Periksa URL, autentikasi, interval, dan pemetaan field."})
    return await request_validation_exception_handler(request, exc)


@app.get("/connection")
def connection_status() -> dict:
    return STATE["source"].status()


@app.put("/connection")
def connection_save(config: SourceConfig) -> dict:
    try:
        return STATE["source"].save(config)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=500, detail="Konfigurasi gagal disimpan; periksa akses folder data") from exc


@app.post("/connection/test")
def connection_test(config: SourceConfig) -> dict:
    try:
        return STATE["source"].preview(config)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/connection/{action}")
def connection_control(action: Literal["start", "stop", "poll"]) -> dict:
    try:
        return getattr(STATE["source"], action)()
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


def _test_frame() -> pd.DataFrame:
    if STATE.get("test") is None:
        df = load_features()
        splits = load_splits()
        STATE["test"] = observed_only(split_frame(df, splits, "test"))
    return STATE["test"]


def _metrics(name: str) -> dict:
    p = METRICS_DIR / name
    if not p.exists():
        raise HTTPException(status_code=503, detail=f"{name} not available; run the pipeline first")
    return json.loads(p.read_text(encoding="utf-8"))


@app.get("/health")
def health() -> dict:
    scorer: AnomalyScorer = STATE["scorer"]
    return {
        "status": "ok",
        "model": scorer.model_tag,
        "threshold": scorer.threshold,
        "n_features": len(scorer.feature_order),
    }


@app.get("/model")
def model_card() -> dict:
    scorer: AnomalyScorer = STATE["scorer"]
    val = _metrics("xgb_final_val.json")
    selection = _metrics("model_selection.json")
    return {
        "model": scorer.model_tag,
        "algorithm": "XGBoost (hist), early stopping on validation aucpr",
        "selected_from": val.get("selected_from"),
        "selection_rule": selection.get("rule"),
        "threshold": scorer.threshold,
        "threshold_source": val.get("threshold_source", "validation split"),
        "validation_metrics": {
            "roc_auc": val.get("roc_auc"),
            "pr_auc": val.get("pr_auc"),
            "brier": val.get("brier"),
            "event_recall": val.get("event_recall"),
        },
        "n_features": len(scorer.feature_order),
        "features": scorer.feature_order,
        "training_data": {
            "dataset": "LEAD1.0-small (ASHRAE GEP III readings + LEAD annotations)",
            "split": "140 train / 30 validation / 30 test buildings (building-disjoint)",
            "observed_readings_only": True,
        },
        "alarm_classes": ["ANOMALY", "OUTAGE", "NORMAL"],
        "caveats": [
            "Model was trained on commercial-building electricity data, not on PLTP Kamojang.",
            "Simulated geothermal signals in this system are labelled SIMULATED.",
        ],
    }


@app.get("/metrics")
def metrics() -> dict:
    return _metrics("holdout_test_final.json")


@app.get("/buildings")
def buildings() -> dict:
    t = _test_frame()
    g = t.groupby(GROUP_COL).agg(n=(TARGET_COL, "size"), n_anomaly=(TARGET_COL, "sum"))

    # Per-building recall from the holdout report, so the dashboard can show how uneven the
    # model's behaviour is across assets (range 0.02-1.00) instead of implying a uniform score.
    recall = {}
    pb = METRICS_DIR.parent / "tables" / "holdout_per_building_final.csv"
    if pb.exists():
        rec = pd.read_csv(pb)
        recall = {int(row["building_id"]): float(row["recall"]) for row in rec.to_dict("records")}

    items = [
        {"building_id": int(b), "n_hours": int(r.n), "n_anomaly": int(r.n_anomaly),
         "recall": recall.get(int(b))}
        for b, r in g.iterrows()
    ]
    values = sorted(v for v in recall.values() if v is not None)
    return {
        "n_buildings": int(len(g)),
        "median_recall": float(pd.Series(values).median()) if values else None,
        "buildings": items,
    }


@app.get("/buildings/{building_id}/series")
def series(building_id: int, limit: int = Query(2000, ge=100, le=9000)) -> dict:
    t = _test_frame()
    d = t[t[GROUP_COL] == building_id]
    if d.empty:
        raise HTTPException(status_code=404, detail=f"building {building_id} not in held-out split")
    d = d.sort_values(TIME_COL)
    if len(d) > limit:  # even stride keeps spikes visible better than tail slicing
        import numpy as np

        idx = np.linspace(0, len(d) - 1, limit).round().astype(int)
        idx = np.unique(idx)
        d = d.iloc[idx]
    scorer: AnomalyScorer = STATE["scorer"]
    from src.data.load import feature_columns

    feats = feature_columns(t)
    p = scorer.model.predict_proba(d[feats])[:, 1]
    return {
        "building_id": building_id,
        "threshold": scorer.threshold,
        "points": [
            {
                "t": r[TIME_COL].strftime("%Y-%m-%dT%H:%M"),
                "v": None if pd.isna(r[VALUE_COL]) else round(float(r[VALUE_COL]), 3),
                "score": round(float(s), 5),
                "y": int(r[TARGET_COL]),
                "flag": int(s >= scorer.threshold),
            }
            for r, s in zip(d.to_dict("records"), p)
        ],
    }


@app.post("/score")
def score(req: ScoreRequest) -> dict:
    scorer: AnomalyScorer = STATE["scorer"]
    hist = pd.DataFrame([r.model_dump() for r in req.history])
    hist.columns = [TIME_COL, VALUE_COL]
    if len(hist) < 24:
        raise HTTPException(status_code=422,
                            detail="at least 24 hourly readings are required to build features")
    try:
        res = scorer.score_history(hist, building_id=req.building_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return res.to_dict()


@app.post("/batch")
def batch(req: BatchRequest) -> dict:
    scorer: AnomalyScorer = STATE["scorer"]
    try:
        df = pd.DataFrame(req.records)
    except Exception as exc:  # pragma: no cover - malformed body
        raise HTTPException(status_code=422, detail=f"invalid records: {exc}") from exc
    missing = {GROUP_COL, TIME_COL, VALUE_COL} - set(df.columns)
    if missing:
        raise HTTPException(status_code=422, detail=f"missing columns: {sorted(missing)}")
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    results = scorer.score_frame(df)
    alarms = [r.to_dict() for r in results]
    return {
        "n_buildings": len(alarms),
        "n_anomaly": sum(1 for a in alarms if a["alarm"] == "ANOMALY"),
        "n_outage": sum(1 for a in alarms if a["alarm"] == "OUTAGE"),
        "results": alarms,
    }


@app.get("/simulate")
def simulate(equipment: str = Query("pump", pattern="^(pump|compressor|bearing|separator)$"),
             limit: int = Query(1200, ge=200, le=2200)) -> dict:
    """SIMULATED geothermal auxiliary-equipment series, scored by the same model.

    Everything returned here is synthetic (D-005). The `warning` field is part of the payload, not
    just documentation, so any client that ignores it is ignoring data it was explicitly given.
    """
    sig_path = SYNTHETIC_DIR / "geothermal_signals_SIMULATED.csv"
    truth_path = SYNTHETIC_DIR / "geothermal_faults_SIMULATED.json"
    if not sig_path.exists():
        raise HTTPException(status_code=503,
                            detail="run `python -m src.simulation.geothermal` first")
    sig = pd.read_csv(sig_path, parse_dates=[TIME_COL])
    sub = sig[sig["equipment"] == equipment].sort_values(TIME_COL)
    if sub.empty:
        raise HTTPException(status_code=404, detail=f"unknown equipment: {equipment}")

    d = sub[["timestamp", "value"]].rename(columns={"value": VALUE_COL}).copy()
    d[GROUP_COL] = 0
    from src.features.build_features import build_feature_frame

    feats = build_feature_frame(d)
    scorer: AnomalyScorer = STATE["scorer"]
    X = feats.reindex(columns=scorer.feature_order)
    score = scorer.model.predict_proba(X)[:, 1]
    ts = feats[TIME_COL]

    faults = json.loads(truth_path.read_text(encoding="utf-8"))["faults"] if truth_path.exists() else []
    mine = [f for f in faults if f["equipment"] == equipment]

    y = pd.Series(0, index=range(len(ts)))
    for f in mine:
        y[(ts >= pd.Timestamp(f["start"])) & (ts <= pd.Timestamp(f["end"]))] = 1

    stride = max(1, len(ts) // limit)
    idx = list(range(0, len(ts), stride))[:limit]
    return {
        "simulated": True,
        "warning": ("SYNTHETIC DATA — not PLTP Kamojang measurements. Generated by "
                    "src/simulation/geothermal.py for demonstration only."),
        "equipment": equipment,
        "unit": str(sub["unit"].iloc[0]),
        "threshold": scorer.threshold,
        "faults": mine,
        "detection": {
            "fault_hours": int(y.sum()),
            "fault_hours_flagged": int(((score >= scorer.threshold) & (y == 1)).sum()),
        },
        "points": [
            {
                "t": ts.iloc[i].strftime("%Y-%m-%dT%H:%M"),
                "v": round(float(sub["value"].iloc[i]), 3),
                "score": round(float(score[i]), 5),
                "fault": int(y.iloc[i]),
            }
            for i in idx
        ],
    }


@app.get("/live/status")
def live_status() -> dict:
    """What the live ring buffers currently hold (no scoring)."""
    return STATE["hub"].status()


@app.post("/live/ingest")
def live_ingest(req: LiveIngestRequest) -> dict:
    """Push real-time readings into the live buffers.

    Readings are appended, not scored here: scoring happens on read (`GET /live/{id}`) so a burst of
    ingest cannot produce a half-built feature window. A null `meter_reading` is accepted and later
    reported as OUTAGE.
    """
    hub = STATE["hub"]
    res = hub.ingest([r.model_dump() for r in req.records])
    return {**res, "buffer": hub.status()}


@app.get("/live/buildings")
def live_buildings() -> dict:
    return STATE["hub"].status()


@app.get("/live/{building_id}")
def live_snapshot(
    building_id: int,
    since: str | None = Query(None, description="ISO timestamp; return only points at/after it"),
    limit: int = Query(500, ge=1, le=2200),
) -> dict:
    """Scored points for one live building, newest last.

    The dashboard polls this with `since` set to the last timestamp it rendered, so each poll
    transfers only the new hours.
    """
    snap = STATE["hub"].snapshot(building_id, since=since, limit=limit)
    if snap["n_points"] == 0 and snap["latest"] is None:
        raise HTTPException(
            status_code=404,
            detail=f"no live readings for building {building_id}; POST /live/ingest first",
        )
    return snap


@app.post("/live/{building_id}/seed")
def live_seed(building_id: int) -> dict:
    """Preload the live buffer for a building from the held-out test split.

    Convenience for a demo: gives the live tail enough history that the 168 h features are defined
    before the first real reading arrives.
    """
    t = _test_frame()
    d = t[t[GROUP_COL] == building_id]
    if d.empty:
        raise HTTPException(status_code=404, detail=f"building {building_id} not in held-out split")
    hist = d[[GROUP_COL, TIME_COL, VALUE_COL]].copy()
    n = STATE["hub"].seed(hist)
    return {"building_id": building_id, "seeded_points": int(n),
            "buffer": STATE["hub"].status()}


# --------------------------------------------------------------------------- imported timelines
@app.get("/datasets")
def datasets_list() -> dict:
    """Imported timeline datasets, newest first."""
    items = ds_store.list_datasets()
    return {"n_datasets": len(items), "datasets": items}


@app.post("/datasets", status_code=201)
def datasets_create(req: DatasetCreate) -> dict:
    """Import a timeline CSV: parse, validate the schema, store it unscored.

    The CSV text is sent in the JSON body rather than as multipart so the dashboard needs no extra
    dependency; the file itself is persisted under data/datasets/<id>/raw.csv.
    """
    import io

    try:
        df = pd.read_csv(io.StringIO(req.csv))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"could not parse CSV: {exc}") from exc
    if df.empty:
        raise HTTPException(status_code=422, detail="CSV contains a header but no data rows")

    try:
        canon, report = ds_store.canonicalize(df)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    ds = ds_store.create_dataset(req.name, canon, report)
    return {"dataset": ds.to_dict(), "import": report}


@app.get("/datasets/{dataset_id}")
def datasets_get(dataset_id: str) -> dict:
    meta = ds_store.get_meta(dataset_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"unknown dataset {dataset_id}")
    scored = ds_store.read_scored(dataset_id)
    return {**meta, "is_scored": scored is not None}


@app.patch("/datasets/{dataset_id}")
def datasets_rename(dataset_id: str, req: DatasetRename) -> dict:
    meta = ds_store.rename_dataset(dataset_id, req.name)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"unknown dataset {dataset_id}")
    return meta


@app.delete("/datasets/{dataset_id}")
def datasets_delete(dataset_id: str) -> dict:
    if not ds_store.delete_dataset(dataset_id):
        raise HTTPException(status_code=404, detail=f"unknown dataset {dataset_id}")
    return {"deleted": dataset_id}


@app.post("/datasets/{dataset_id}/score")
def datasets_score(dataset_id: str, req: DatasetScoreRequest | None = None) -> dict:
    """Score an imported timeline and cache the result.

    Scoring a long history is CPU work, so the result is written to parquet and served afterwards by
    `/datasets/{id}/series`; re-scoring only happens when asked.
    """
    meta = ds_store.get_meta(dataset_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"unknown dataset {dataset_id}")
    raw = ds_store.read_raw(dataset_id)
    if req is not None and req.building_id is not None:
        raw = raw[raw[GROUP_COL] == req.building_id]
        if raw.empty:
            raise HTTPException(
                status_code=404, detail=f"building {req.building_id} not in this dataset"
            )

    scorer: AnomalyScorer = STATE["scorer"]
    scored = ds_store.score_frame_long(scorer, raw)
    if scored.empty:
        raise HTTPException(
            status_code=422,
            detail="no row could be scored; each building needs enough consecutive hours of history",
        )
    summary = ds_store.summarize_scored(scored, scorer.threshold)
    ds_store.save_scored(dataset_id, scored, summary)
    return {
        "dataset_id": dataset_id,
        "name": meta["name"],
        "model": scorer.model_tag,
        "scored_rows": int(len(scored)),
        "summary": summary,
        "series_url": f"/datasets/{dataset_id}/series",
    }


@app.get("/datasets/{dataset_id}/series")
def datasets_series(
    dataset_id: str,
    building_id: int | None = Query(None),
    limit: int = Query(2000, ge=100, le=20000),
) -> dict:
    """Scored points from a previously scored dataset, newest last."""
    if ds_store.get_meta(dataset_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown dataset {dataset_id}")
    scored = ds_store.read_scored(dataset_id)
    if scored is None:
        raise HTTPException(
            status_code=409, detail="dataset not scored yet; POST /datasets/{id}/score first"
        )
    d = scored if building_id is None else scored[scored[GROUP_COL] == building_id]
    if d.empty:
        raise HTTPException(status_code=404, detail="no rows for the requested building")

    if len(d) > limit:  # even stride keeps isolated spikes visible better than tail slicing
        import numpy as np

        idx = np.unique(np.linspace(0, len(d) - 1, limit).round().astype(int))
        d = d.iloc[idx]

    return {
        "dataset_id": dataset_id,
        "building_id": building_id,
        "threshold": float(STATE["scorer"].threshold),
        "n_rows": int(len(d)),
        "points": [
            {
                "t": r[TIME_COL].strftime("%Y-%m-%dT%H:%M"),
                "v": None if pd.isna(r[VALUE_COL]) else round(float(r[VALUE_COL]), 3),
                "score": None if pd.isna(r["score"]) else round(float(r["score"]), 5),
                "alarm": r["alarm"],
                "flag": int(r["flag"]),
                "building_id": int(r[GROUP_COL]),
            }
            for r in d.to_dict("records")
        ],
    }


@app.get("/datasets/{dataset_id}/anomalies")
def datasets_anomalies(
    dataset_id: str,
    limit: int = Query(100, ge=1, le=2000),
    building_id: int | None = Query(None),
) -> dict:
    """Flagged hours only, worst first -- feeds the anomaly table without shipping the whole series."""
    if ds_store.get_meta(dataset_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown dataset {dataset_id}")
    scored = ds_store.read_scored(dataset_id)
    if scored is None:
        raise HTTPException(
            status_code=409, detail="dataset not scored yet; POST /datasets/{id}/score first"
        )
    d = scored[scored["alarm"] == "ANOMALY"]
    if building_id is not None:
        d = d[d[GROUP_COL] == building_id]
    d = d.sort_values("score", ascending=False).head(limit)
    return {
        "dataset_id": dataset_id,
        "n_anomaly_total": int((scored["alarm"] == "ANOMALY").sum()),
        "n_returned": int(len(d)),
        "events": [
            {
                "building_id": int(r[GROUP_COL]),
                "t": r[TIME_COL].strftime("%Y-%m-%dT%H:%M"),
                "v": None if pd.isna(r[VALUE_COL]) else round(float(r[VALUE_COL]), 3),
                "score": round(float(r["score"]), 5),
            }
            for r in d.to_dict("records")
        ],
    }


@app.get("/")
def dashboard() -> FileResponse:
    index = STATIC_DIR / "index.html"
    if not index.exists():
        return JSONResponse({"detail": "dashboard not built"}, status_code=404)
    return FileResponse(index)


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
