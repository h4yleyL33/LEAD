"""One configurable REST/JSON source feeding the existing live hub."""
from __future__ import annotations

import ipaddress
import json
import math
import re
import socket
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from urllib.parse import parse_qsl, urlsplit

import httpx
from pydantic import BaseModel, Field, SecretStr, model_validator


class SourceConfig(BaseModel):
    name: str = Field(default="Perangkat IoT", min_length=1, max_length=80)
    url: str = Field(min_length=8, max_length=2000)
    interval_seconds: int = Field(default=30, ge=5, le=3600)
    allow_local_network: bool = False
    simulated: bool = False
    auth: Literal["none", "bearer", "api_key"] = "none"
    header_name: str = Field(default="X-API-Key", pattern=r"^[A-Za-z0-9-]{1,80}$")
    token: SecretStr | None = Field(default=None, max_length=4096)
    records_path: str = Field(default="records", max_length=160)
    building_field: str = Field(default="building_id", max_length=160)
    fixed_building_id: int = Field(default=1, ge=0, le=2**63-1)
    timestamp_field: str = Field(default="timestamp", min_length=1, max_length=160)
    timestamp_format: Literal["iso", "unix_s", "unix_ms"] = "iso"
    value_field: str = Field(default="meter_reading", min_length=1, max_length=160)

    @model_validator(mode="after")
    def validate_destination(self):
        try:
            u = urlsplit(self.url)
            port = u.port
        except ValueError as exc:
            raise ValueError("URL atau port tidak valid") from exc
        if u.scheme not in {"http", "https"} or not u.hostname or u.username or u.password or u.fragment:
            raise ValueError("Gunakan URL HTTP/HTTPS tanpa kredensial atau fragmen")
        if u.scheme == "http" and not self.allow_local_network:
            raise ValueError("API internet harus memakai HTTPS")
        if self.auth != "none" and u.scheme != "https":
            raise ValueError("Autentikasi token memerlukan HTTPS")
        if self.header_name.lower() in {"host", "content-length", "connection", "transfer-encoding", "cookie"}:
            raise ValueError("Nama header autentikasi tidak diizinkan")
        if any(re.search(r"key|token|secret|password|signature|auth", k, re.I) for k, _ in parse_qsl(u.query)):
            raise ValueError("Masukkan kredensial lewat header, bukan URL")
        if self.token and any(c in self.token.get_secret_value() for c in "\r\n"):
            raise ValueError("Token tidak boleh mengandung baris baru")
        return self


def validate_addresses(config: SourceConfig) -> None:
    u = urlsplit(config.url)
    try:
        addresses = {ipaddress.ip_address(info[4][0]) for info in
                     socket.getaddrinfo(u.hostname, u.port or (443 if u.scheme == "https" else 80),
                                        type=socket.SOCK_STREAM)}
    except (OSError, ValueError) as exc:
        raise ValueError("Alamat API tidak dapat ditemukan") from exc
    for address in addresses:
        if address.is_link_local or address.is_multicast or address.is_unspecified or address.is_reserved:
            raise ValueError("Alamat tujuan tidak diizinkan")
        if not address.is_global and not (config.allow_local_network and
                                          (address.is_private or address.is_loopback)):
            raise ValueError("Aktifkan izin jaringan lokal untuk alamat perangkat ini")
        if u.scheme == "http" and address.is_global:
            raise ValueError("API internet harus memakai HTTPS")


def field_at(value, path: str):
    for key in path.split(".") if path else []:
        if not isinstance(value, dict) or key not in value:
            raise ValueError(f"Field {path!r} tidak ditemukan")
        value = value[key]
    return value


def map_records(payload, config: SourceConfig) -> list[dict]:
    rows = field_at(payload, config.records_path)
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list) or not rows or len(rows) > 10000:
        raise ValueError("Path data harus berisi 1–10000 objek pembacaan")
    records = {}
    for i, row in enumerate(rows, 1):
        try:
            bid = field_at(row, config.building_field) if config.building_field else config.fixed_building_id
            if isinstance(bid, bool) or not re.fullmatch(r"\d+", str(bid)) or int(bid) > 2**63-1:
                raise ValueError("ID aset harus berupa bilangan bulat positif atau nol")
            raw_time = field_at(row, config.timestamp_field)
            if config.timestamp_format == "iso":
                ts = datetime.fromisoformat(str(raw_time).replace("Z", "+00:00"))
                ts = ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)
            else:
                ts = datetime.fromtimestamp(float(raw_time) / (1000 if config.timestamp_format == "unix_ms" else 1), timezone.utc)
            if ts.minute or ts.second or ts.microsecond:
                raise ValueError("Data harus per jam pada menit 00; agregasikan data sensor terlebih dahulu")
            value = field_at(row, config.value_field)
            if value is not None:
                if isinstance(value, bool):
                    raise ValueError("Reading harus numerik atau null")
                value = float(value)
                if not math.isfinite(value) or value < 0:
                    raise ValueError("Reading kWh harus berhingga dan tidak negatif")
            record = {"building_id": int(bid), "timestamp": ts.replace(tzinfo=None).isoformat(),
                      "meter_reading": value}
            records[(int(bid), record["timestamp"])] = record
        except (ValueError, TypeError, OverflowError, OSError) as exc:
            # Do not echo a raw value: upstream fields may contain credentials or private data.
            message = str(exc) if isinstance(exc, ValueError) and str(exc).startswith(("Field ", "ID aset", "Data harus", "Reading")) else "Format waktu atau angka tidak valid"
            raise ValueError(f"Baris {i}: {message}") from exc
    return sorted(records.values(), key=lambda r: (r["timestamp"], r["building_id"]))


def fetch_records(config: SourceConfig, token: str | None) -> list[dict]:
    validate_addresses(config)
    headers = {"Accept": "application/json"}
    if config.auth != "none":
        if not token:
            raise ValueError("Isi token autentikasi terlebih dahulu")
        headers["Authorization" if config.auth == "bearer" else config.header_name] = (
            f"Bearer {token}" if config.auth == "bearer" else token)
    # shortcut: DNS is checked before HTTPX resolves; pin addresses if exposed beyond localhost.
    try:
        with httpx.Client(timeout=httpx.Timeout(5, connect=3), follow_redirects=False, trust_env=False) as client:
            started = time.monotonic()
            with client.stream("GET", config.url, headers=headers) as response:
                if response.status_code != 200:
                    raise ValueError(f"API sumber mengembalikan HTTP {response.status_code}")
                body = bytearray()
                for chunk in response.iter_bytes(chunk_size=65536):
                    body.extend(chunk)
                    if len(body) > 2_000_000 or time.monotonic() - started > 15:
                        raise ValueError("Respons API melebihi batas 2 MB atau 15 detik")
                try:
                    payload = json.loads(body)
                except (ValueError, UnicodeError, RecursionError) as exc:
                    raise ValueError("Respons API bukan JSON yang valid") from exc
        return map_records(payload, config)
    except httpx.HTTPError as exc:
        raise ValueError("Koneksi ke API gagal atau kehabisan waktu; periksa URL, jaringan, dan TLS") from exc


class ExternalSource:
    def __init__(self, hub, config_path: Path):
        self.hub = hub
        self.path = config_path
        self.config: SourceConfig | None = None
        self.token: str | None = None
        self.error = ""
        self.last_poll = None
        self.last_success = None
        self.last_received = 0
        self.last_added = 0
        self.total_added = 0
        self.seen: dict[tuple, float | None] = {}
        self.lock = threading.RLock()
        self.poll_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        if self.path.exists():
            try:
                self.config = SourceConfig.model_validate_json(self.path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                self.error = "Konfigurasi tersimpan tidak dapat dibaca; simpan ulang koneksi"

    def status(self) -> dict:
        with self.lock:
            return {"config": self.config.model_dump(exclude={"token"}) if self.config else None,
                    "has_token": bool(self.token), "running": bool(self.thread and self.thread.is_alive()),
                    "last_poll": self.last_poll, "last_success": self.last_success,
                    "last_received": self.last_received, "last_added": self.last_added,
                    "total_added": self.total_added, "error": self.error}

    def effective_token(self, config: SourceConfig) -> str | None:
        if config.auth == "none":
            return None
        if config.token and config.token.get_secret_value():
            return config.token.get_secret_value()
        if self.config and (config.url, config.auth, config.header_name) == (
                self.config.url, self.config.auth, self.config.header_name):
            return self.token
        return None

    def save(self, config: SourceConfig) -> dict:
        with self.poll_lock, self.lock:
            if self.thread and self.thread.is_alive():
                raise ValueError("Hentikan koneksi sebelum mengubah konfigurasi")
            token = self.effective_token(config)
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temp = self.path.with_suffix(".tmp")
            temp.write_text(json.dumps(config.model_dump(exclude={"token"}), indent=2), encoding="utf-8")
            temp.replace(self.path)
            self.config = config.model_copy(update={"token": None})
            self.token = token
            self.error = ""
            self.seen.clear()
            self.last_poll = self.last_success = None
            self.last_received = self.last_added = self.total_added = 0
            return self.status()

    def preview(self, config: SourceConfig) -> dict:
        with self.lock:
            token = self.effective_token(config)
        records = fetch_records(config, token)
        return {"count": len(records), "preview": records[:5]}

    def poll(self) -> dict:
        if not self.poll_lock.acquire(blocking=False):
            raise ValueError("Pengambilan sebelumnya masih berjalan")
        try:
            with self.lock:
                config, token = self.config, self.token
                self.last_poll = datetime.now(timezone.utc).isoformat()
            if config is None:
                raise ValueError("Simpan konfigurasi koneksi terlebih dahulu")
            records = fetch_records(config, token)
            new = [r for r in records if (r["building_id"], r["timestamp"]) not in self.seen or
                   self.seen[(r["building_id"], r["timestamp"])] != r["meter_reading"]]
            if new:
                self.hub.ingest(new)
            with self.lock:
                for r in new:
                    self.seen[(r["building_id"], r["timestamp"])] = r["meter_reading"]
                while len(self.seen) > 10000:
                    self.seen.pop(next(iter(self.seen)))
                self.last_success = datetime.now(timezone.utc).isoformat()
                self.last_received, self.last_added = len(records), len(new)
                self.total_added += len(new)
                self.error = ""
        except ValueError as exc:
            with self.lock:
                self.error = str(exc)
                self.last_added = 0
            raise
        finally:
            self.poll_lock.release()
        return self.status()

    def _run(self):
        while not self.stop_event.is_set():
            try:
                self.poll()
            except ValueError:
                pass
            except Exception:
                with self.lock:
                    self.error = "Pengambilan data gagal; hentikan koneksi dan coba lagi"
            if self.stop_event.wait(self.config.interval_seconds):
                break

    def start(self) -> dict:
        with self.lock:
            if not self.config:
                raise ValueError("Simpan konfigurasi koneksi terlebih dahulu")
            if self.config.auth != "none" and not self.token:
                raise ValueError("Isi ulang token setelah server restart")
            if not (self.thread and self.thread.is_alive()):
                self.stop_event.clear()
                self.thread = threading.Thread(target=self._run, name="iot-api-source", daemon=True)
                self.thread.start()
            return self.status()

    def stop(self) -> dict:
        self.stop_event.set()
        thread = self.thread
        if thread:
            thread.join(timeout=20)
        return self.status()
