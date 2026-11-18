from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from pydantic import ValidationError

from backend.app.external_api import ExternalSource, SourceConfig, map_records


@pytest.fixture
def source_api():
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "/readings")
                self.end_headers()
                return
            body = b"x" * 2_000_001 if self.path == "/large" else json.dumps({
                "data": {"readings": [
                    {"device": {"id": 9001}, "at": "2026-11-18T07:00:00+07:00", "kwh": "12.5"},
                    {"device": {"id": 9001}, "at": "2026-11-18T01:00:00Z", "kwh": None},
                ]}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()
    thread.join()


def config(url, **changes):
    return SourceConfig(url=url, allow_local_network=True, records_path="data.readings",
                        building_field="device.id", timestamp_field="at", value_field="kwh", **changes)


class Hub:
    def __init__(self):
        self.records = []

    def ingest(self, records):
        self.records.extend(records)


def test_fetch_map_and_deduplicate(source_api, tmp_path):
    hub = Hub()
    source = ExternalSource(hub, tmp_path / "source.json")
    cfg = config(source_api + "/readings")
    preview = source.preview(cfg)
    assert preview["count"] == 2
    assert not hub.records
    assert preview["preview"][0] == {
        "building_id": 9001, "timestamp": "2026-11-18T00:00:00", "meter_reading": 12.5}
    source.save(cfg)
    assert source.poll()["last_added"] == 2
    assert source.poll()["last_added"] == 0
    assert len(hub.records) == 2
    assert hub.records[1]["meter_reading"] is None
    loaded = ExternalSource(hub, source.path)
    assert loaded.status()["config"]["url"] == cfg.url
    assert not loaded.status()["running"]


def test_background_runs_without_browser_and_stops(source_api, tmp_path):
    source = ExternalSource(Hub(), tmp_path / "source.json")
    source.save(config(source_api + "/readings"))
    source.start()
    try:
        deadline = time.monotonic() + 3
        while not source.status()["last_success"] and time.monotonic() < deadline:
            time.sleep(.02)
        assert source.status()["last_added"] == 2
        with pytest.raises(ValueError, match="Hentikan"):
            source.save(config(source_api + "/readings"))
    finally:
        assert not source.stop()["running"]


def test_secret_never_returned_or_persisted(tmp_path):
    source = ExternalSource(Hub(), tmp_path / "source.json")
    cfg = SourceConfig(url="https://example.com/data", auth="bearer", token="private-test-token")
    status = source.save(cfg)
    assert status["has_token"]
    assert "private-test-token" not in json.dumps(status)
    assert "private-test-token" not in source.path.read_text()
    assert source.effective_token(SourceConfig(url=cfg.url, auth="bearer")) == "private-test-token"
    assert source.effective_token(SourceConfig(url="https://different.example/data", auth="bearer")) is None
    reloaded = ExternalSource(Hub(), source.path)
    with pytest.raises(ValueError, match="token"):
        reloaded.start()


@pytest.mark.parametrize("url", ["file:///etc/passwd", "http://example.com", "https://u:p@example.com", "https://example.com?api_key=secret"])
def test_reject_unsafe_url(url):
    with pytest.raises(ValidationError):
        SourceConfig(url=url)


def test_local_requires_opt_in_and_blocks_metadata(tmp_path):
    source = ExternalSource(Hub(), tmp_path / "source.json")
    with pytest.raises(ValueError, match="jaringan lokal"):
        source.preview(SourceConfig(url="https://127.0.0.1/readings"))
    with pytest.raises(ValueError, match="tidak diizinkan"):
        source.preview(config("http://169.254.169.254/"))


def test_redirect_and_oversize_are_rejected(source_api, tmp_path):
    source = ExternalSource(Hub(), tmp_path / "source.json")
    with pytest.raises(ValueError, match="302"):
        source.preview(config(source_api + "/redirect"))
    with pytest.raises(ValueError, match="2 MB"):
        source.preview(config(source_api + "/large"))


def test_bad_mapping_is_atomic_and_hourly(tmp_path):
    cfg = SourceConfig(url="https://example.com", building_field="", fixed_building_id=12)
    with pytest.raises(ValueError, match="Baris 2"):
        map_records({"records": [
            {"timestamp": "2026-11-18T00:00:00Z", "meter_reading": 12},
            {"timestamp": "2026-11-18T00:01:00Z", "meter_reading": 15},
        ]}, cfg)
    with pytest.raises(ValueError, match="tidak negatif"):
        map_records({"records": [{"timestamp": "2026-11-18T00:00:00", "meter_reading": -1}]}, cfg)
    epoch = cfg.model_copy(update={"timestamp_format": "unix_ms"})
    result = map_records({"records": [{"timestamp": 0, "meter_reading": "0"}]}, epoch)
    assert result[0]["building_id"] == 12
    assert result[0]["timestamp"] == "1970-01-01T00:00:00"


def test_poll_failure_preserves_buffer_and_recovers(source_api, tmp_path):
    hub = Hub()
    source = ExternalSource(hub, tmp_path / "source.json")
    source.save(config(source_api + "/redirect"))
    with pytest.raises(ValueError):
        source.poll()
    assert source.status()["error"]
    assert not hub.records
    source.save(config(source_api + "/readings"))
    assert source.poll()["last_added"] == 2
    assert not source.status()["error"]


@pytest.mark.parametrize("auth,header,expected", [("bearer", "Authorization", "Bearer test-secret"),
                                                 ("api_key", "X-Sensor-Key", "test-secret")])
def test_auth_header_and_sanitized_http_errors(auth, header, expected, tmp_path, monkeypatch):
    import httpx
    from backend.app import external_api

    def respond(request):
        assert request.headers[header] == expected
        return httpx.Response(401, text="private upstream error with test-secret")

    original_client = httpx.Client
    monkeypatch.setattr(external_api, "validate_addresses", lambda config: None)
    monkeypatch.setattr(external_api.httpx, "Client", lambda **kwargs: original_client(
        transport=httpx.MockTransport(respond), **kwargs))
    source = ExternalSource(Hub(), tmp_path / "source.json")
    cfg = SourceConfig(url="https://example.com", auth=auth, header_name="X-Sensor-Key", token="test-secret")
    with pytest.raises(ValueError, match="HTTP 401") as failure:
        source.preview(cfg)
    assert "test-secret" not in str(failure.value)


def test_api_routes_validate_and_hide_secrets(source_api, tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from backend.app import main

    hub = Hub()
    source = ExternalSource(hub, tmp_path / "source.json")
    monkeypatch.setitem(main.STATE, "source", source)
    client = TestClient(main.app)
    cfg = config(source_api + "/readings").model_dump(mode="json")
    response = client.post("/connection/test", json=cfg)
    assert response.status_code == 200
    assert response.json()["count"] == 2
    assert not hub.records
    assert client.put("/connection", json=cfg).status_code == 200
    assert client.post("/connection/poll").json()["last_added"] == 2
    assert client.get("/connection").json()["total_added"] == 2
    assert not client.post("/connection/stop").json()["running"]
    invalid = {**cfg, "auth": "bearer", "token": "must-not-echo"}
    response = client.post("/connection/test", json=invalid)
    assert response.status_code == 422
    assert "must-not-echo" not in response.text


def test_invalid_json_and_mapping_keep_existing_data(source_api, tmp_path, monkeypatch):
    from backend.app import external_api

    hub = Hub()
    hub.records = [{"existing": True}]
    source = ExternalSource(hub, tmp_path / "source.json")
    source.save(config(source_api + "/readings"))
    monkeypatch.setattr(external_api, "fetch_records", lambda *args: (_ for _ in ()).throw(ValueError("bad JSON")))
    with pytest.raises(ValueError, match="bad JSON"):
        source.poll()
    assert hub.records == [{"existing": True}]


def test_live_buffer_remains_bounded():
    from backend.app.live import BUFFER_HOURS, LiveHub
    hub = LiveHub(None)
    hub.ingest([{"building_id": 9001, "timestamp": "2026-11-18T00:00:00", "meter_reading": 1}] * (BUFFER_HOURS + 5))
    assert hub.status()["n_points"] == BUFFER_HOURS


@pytest.mark.parametrize("bad", [float("inf"), -1, True])
def test_invalid_reading_is_rejected(bad):
    cfg = SourceConfig(url="https://example.com")
    with pytest.raises(ValueError):
        map_records({"records": [{"building_id": 0, "timestamp": "2026-11-18T00:00:00", "meter_reading": bad}]}, cfg)
