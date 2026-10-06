"""Local browser viewer for saved attempts and aggregate time-to-target results.

Open / for per-attempt trajectories, verdicts and GPU samples, or /results for
intervention comparisons and measured time to target. The server reads an
attempts directory (default ``$SPEEDRUN_HOME/attempts`` or ./attempts), makes no
inference calls and binds to 127.0.0.1 by default:

    python -m reasoning_speedrun.viewer --attempts examples/attempts
"""
from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlparse

from reasoning_speedrun.lib.common import workdir
from reasoning_speedrun.viewer.results import build_results
from reasoning_speedrun.viewer.store import AttemptStore

STATIC = Path(__file__).resolve().parent
HTML, CSS, JS = "text/html; charset=utf-8", "text/css; charset=utf-8", "text/javascript; charset=utf-8"
ASSETS = {
    "/": ("attempts_ui/index.html", HTML),
    "/attempts": ("attempts_ui/index.html", HTML),
    "/attempts/viewer.css": ("attempts_ui/viewer.css", CSS),
    "/attempts/viewer.js": ("attempts_ui/viewer.js", JS),
    "/results": ("results_ui/index.html", HTML),
    "/results/": ("results_ui/index.html", HTML),
    "/results/viewer.css": ("results_ui/viewer.css", CSS),
    "/results/viewer.js": ("results_ui/viewer.js", JS),
    "/shared.css": ("shared.css", CSS),
}


def make_handler(store: AttemptStore):
    class Handler(BaseHTTPRequestHandler):
        def send_bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, value: object, status: int = 200) -> None:
            self.send_bytes(json.dumps(value, ensure_ascii=False).encode(), "application/json; charset=utf-8", status)

        def do_GET(self) -> None:
            path = unquote(urlparse(self.path).path)
            if path in ASSETS:
                filename, content_type = ASSETS[path]
                self.send_bytes((STATIC / filename).read_bytes(), content_type)
                return
            try:
                if path == "/api/results":
                    self.send_json(build_results(store))
                    return
                if path == "/api/attempts":
                    self.send_json(store.list())
                    return
                if match := re.fullmatch(r"/api/attempts/([^/]+)/overview", path):
                    self.send_json(store.overview(match[1]))
                    return
                if match := re.fullmatch(r"/api/attempts/([^/]+)/gpu", path):
                    self.send_json(store.gpu(match[1]))
                    return
                if match := re.fullmatch(r"/api/attempts/([^/]+)/questions/(\d+)", path):
                    self.send_json(store.question(match[1], int(match[2])))
                    return
                if match := re.fullmatch(r"/api/attempts/([^/]+)/questions/(\d+)/rollouts/(\d+)", path):
                    self.send_json(store.rollout(match[1], int(match[2]), int(match[3])))
                    return
                if match := re.fullmatch(r"/api/attempts/([^/]+)/files/(.+)", path):
                    artifact = store.artifact(match[1], match[2])
                    self.send_bytes(
                        artifact.read_bytes(),
                        "application/json; charset=utf-8" if artifact.suffix == ".json" else "application/x-ndjson; charset=utf-8",
                    )
                    return
            except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError) as exc:
                self.send_json({"error": str(exc)}, 404)
                return
            self.send_json({"error": "Not found"}, 404)

        def log_message(self, format: str, *args: object) -> None:
            print(f"{self.address_string()} - {format % args}")

    return Handler


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--attempts", type=Path, default=workdir() / "attempts", help="Directory of attempt folders")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if not args.attempts.is_dir():
        parser.error(f"Attempts directory not found: {args.attempts}")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(AttemptStore(args.attempts.resolve())))
    print(f"Attempt viewer: http://{args.host}:{args.port} (attempts: {args.attempts})", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
