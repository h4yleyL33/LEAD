"""API tests. Run with: python -m pytest backend/tests -q

The API is exercised through the real application object, with the real model artefact, so a
broken model file or a schema change fails here rather than in the demo.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.main import app  # noqa: E402
from src.utils.config import MODELS_DIR  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (MODELS_DIR / "xgb_final.json").exists(),
    reason="model artefact missing; run the training pipeline first",
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health(client) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"] == "xgboost_final"
    assert 0.0 < body["threshold"] < 1.0
    assert body["n_features"] > 0


def test_model_card(client) -> None:
    r = client.get("/model")
    assert r.status_code == 200
    b = r.json()
    assert b["validation_metrics"]["roc_auc"] > 0.5
    assert b["threshold_source"].startswith("val")
    assert any("SIMULATED" in c for c in b["caveats"])


def test_metrics_match_published_number(client) -> None:
    r = client.get("/metrics")
    assert r.status_code == 200
    m = r.json()
    # The number the paper quotes must be the number the API serves.
    assert m["roc_auc"] == pytest.approx(0.9524, abs=5e-4)
    assert m["split_evaluated"] == "test"
    assert m["n_buildings"] == 30


def test_buildings_and_series(client) -> None:
    r = client.get("/buildings")
    assert r.status_code == 200
    b = r.json()
    assert b["n_buildings"] == 30
    bid = b["buildings"][0]["building_id"]

    r2 = client.get(f"/buildings/{bid}/series", params={"limit": 500})
    assert r2.status_code == 200
    s = r2.json()
    assert 100 <= len(s["points"]) <= 500
    assert all(0.0 <= p["score"] <= 1.0 for p in s["points"])
    assert all(p["flag"] in (0, 1) for p in s["points"])


def test_series_unknown_building_404(client) -> None:
    assert client.get("/buildings/999999/series").status_code == 404


def test_score_endpoint(client) -> None:
    s = client.get("/buildings").json()["buildings"][0]["building_id"]
    pts = client.get(f"/buildings/{s}/series", params={"limit": 400}).json()["points"]
    history = [{"timestamp": p["t"], "meter_reading": p["v"]} for p in pts]
    r = client.post("/score", json={"building_id": s, "history": history})
    assert r.status_code == 200
    body = r.json()
    assert body["alarm"] in {"ANOMALY", "NORMAL", "OUTAGE"}
    assert 0.0 <= body["score"] <= 1.0
    assert body["model"] == "xgboost_final"


def test_score_rejects_short_history(client) -> None:
    r = client.post("/score", json={"building_id": 1,
                                    "history": [{"timestamp": "2016-06-01T00:00",
                                                 "meter_reading": 10.0}]})
    assert r.status_code == 422


def test_score_reports_outage(client) -> None:
    s = client.get("/buildings").json()["buildings"][0]["building_id"]
    pts = client.get(f"/buildings/{s}/series", params={"limit": 400}).json()["points"]
    history = [{"timestamp": p["t"], "meter_reading": p["v"]} for p in pts]
    history[-1]["meter_reading"] = None
    r = client.post("/score", json={"building_id": s, "history": history})
    assert r.status_code == 200
    assert r.json()["alarm"] == "OUTAGE"


def test_batch_endpoint(client) -> None:
    ids = [b["building_id"] for b in client.get("/buildings").json()["buildings"][:2]]
    records = []
    for bid in ids:
        pts = client.get(f"/buildings/{bid}/series", params={"limit": 400}).json()["points"]
        records += [{"building_id": bid, "timestamp": p["t"], "meter_reading": p["v"]} for p in pts]
    r = client.post("/batch", json={"records": records})
    assert r.status_code == 200
    body = r.json()
    assert body["n_buildings"] == 2
    assert len(body["results"]) == 2


def test_batch_validates_columns(client) -> None:
    r = client.post("/batch", json={"records": [{"foo": 1}]})
    assert r.status_code == 422


def test_dashboard_is_served(client) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "SIMULATED" in r.text  # the disclaimer must be visible on the page itself


def test_simulate_endpoint_is_labelled_simulated(client) -> None:
    r = client.get("/simulate", params={"equipment": "pump", "limit": 400})
    assert r.status_code == 200
    b = r.json()
    assert b["simulated"] is True
    assert "not PLTP Kamojang" in b["warning"]
    assert len(b["points"]) > 0
    assert all(0.0 <= p["score"] <= 1.0 for p in b["points"])
    assert all(p["fault"] in (0, 1) for p in b["points"])


def test_simulate_rejects_unknown_equipment(client) -> None:
    assert client.get("/simulate", params={"equipment": "turbine"}).status_code == 422
