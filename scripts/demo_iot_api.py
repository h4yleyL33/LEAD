"""Local REST source for testing Koneksi API; every reading is SIMULATED."""
import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/readings":
            self.send_error(404)
            return
        end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        records = [{"device_id": 9001,
                    "time": (end - timedelta(hours=95-i)).isoformat(),
                    "energy_kwh": round(100 + 20 * math.sin(i / 4) + (100 if i == 90 else 0), 2)}
                   for i in range(96)]
        body = json.dumps({"simulated": True, "data": {"readings": records}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    print(f"SIMULATED IoT API: http://127.0.0.1:{args.port}/readings", flush=True)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
