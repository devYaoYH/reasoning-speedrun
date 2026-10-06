"""Minimal rate-limited math-grader oracle (HTTP, stdlib only).

ONE global FIFO gate: the server answers at most one /verify request every
`cost_c` seconds, across ALL clients and ALL processes (one queue, one worker,
one clock in one process). Grading is instant; the delay is the artificial cost.

Run:   $PY server.py            # reads ./config.yaml
       GRADER_CONFIG=foo.yaml $PY server.py   # alternate config (used by tests)

Endpoints:
  POST /verify   {"index": int, "candidate": str, "agent_id"?: str, "query_id"?: str}
                 -> {"query_id": str, "seq": int, "agent_id": str|null,
                     "index": int, "candidate": str, "verdict": bool,
                     "queue_ahead_at_submit": int, "submitted_at": iso,
                     "picked_at": iso, "answered_at": iso,
                     "queue_wait_s": float, "toll_s": float, "latency_s": float}
                 (gold answer is NOT returned — only the verdict; it is logged.)
  GET  /health   -> {"ok": true, "n_problems": int, "index_min": int,
                     "index_max": int, "cost_c": float, "queries_so_far": int}

Audit log (one JSON line per answered query) = the /verify response PLUS "gold".
"""
import os
import hashlib
import sys
import json
import time
import uuid
import queue
import threading
import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "grader_core"))
from grade import grade  # noqa: E402


def iso(epoch):
    """Format an epoch float as a UTC ISO-8601 timestamp."""
    return datetime.datetime.fromtimestamp(
        epoch, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _load_secrets():
    """Best-effort: load HF_TOKEN etc. from repo-root `secrets` (never printed)."""
    secrets = os.path.join(os.path.dirname(HERE), "secrets")
    if os.path.isfile(secrets):
        for line in open(secrets):
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())
    tok = os.environ.get("HF_TOKEN")
    if tok:
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", tok)


def load_gold(cfg):
    """Return {index(int): gold_str}. Supports hf | jsonl | csv | parquet."""
    d = cfg["dataset"]
    idx_f = d.get("idx_field", "problem_idx")
    gold_f = d.get("gold_field", "answer")
    fmt = d["format"]
    src = d["source"]
    if fmt != "hf" and not os.path.isabs(src):
        src = os.path.join(HERE, src)   # resolve local files against the tool dir, not CWD
    if fmt == "hf":
        from datasets import load_dataset
        ds = load_dataset(src, token=os.environ.get("HF_TOKEN"))["train"]
        rows = list(ds)
    elif fmt == "jsonl":
        with open(src) as f:
            rows = [json.loads(line) for line in f if line.strip()]
    elif fmt == "csv":
        import csv
        with open(src) as f:
            rows = list(csv.DictReader(f))
    elif fmt == "parquet":
        import pandas as pd
        rows = pd.read_parquet(src).to_dict("records")
    else:
        raise ValueError(f"unknown dataset format: {fmt}")
    gold = {int(r[idx_f]): str(r[gold_f]) for r in rows}
    if not gold:
        raise ValueError("no problems loaded from dataset")
    return gold


class Oracle:
    """The single global gate: one FIFO queue, one worker.

    TOLL MODEL: every query costs >= cost_c, charged from when the worker picks it
    up (as if a human grader needs cost_c to check it). Because the worker is serial
    and each job takes >= cost_c, consecutive answers are automatically >= cost_c
    apart and the FIRST query is charged just like the rest.
    """

    def __init__(self, gold, cost_c, audit_path, dataset=None):
        self.dataset = dataset or {}
        self.gold = gold
        self.cost_c = float(cost_c)
        self.audit_path = audit_path
        self.q = queue.Queue()
        self.seq = 0
        worker = threading.Thread(target=self._worker, daemon=True)
        worker.start()

    def _worker(self):
        while True:
            job = self.q.get()
            picked = time.time()                       # toll starts at pickup
            gold = self.gold.get(job["index"])
            verdict = grade(job["candidate"], gold)
            wait = self.cost_c - (time.time() - picked)  # grading takes >= cost_c
            if wait > 0:
                time.sleep(wait)
            answered = time.time()
            self.seq += 1
            # Full audit record — everything needed to read the log standalone.
            record = {
                "dataset": self.dataset,
                "query_id": job["query_id"],
                "seq": self.seq,
                "agent_id": job.get("agent_id"),
                "index": job["index"],
                "candidate": job["candidate"],
                "gold": gold,
                "verdict": bool(verdict),
                "queue_ahead_at_submit": job["queue_ahead"],
                "submitted_at": iso(job["submitted_epoch"]),
                "picked_at": iso(picked),
                "answered_at": iso(answered),
                "queue_wait_s": round(picked - job["submitted_epoch"], 3),
                "toll_s": round(answered - picked, 3),
                "latency_s": round(answered - job["submitted_epoch"], 3),
            }
            try:
                with open(self.audit_path, "a") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception:
                pass
            # Client response = the full record MINUS gold (the oracle returns a
            # verdict, never the answer itself).
            job["result"] = {k: v for k, v in record.items() if k != "gold"}
            job["done"].set()

    def submit(self, index, candidate, agent_id=None, query_id=None):
        job = {
            "query_id": query_id or uuid.uuid4().hex,
            "index": index,
            "candidate": candidate,
            "agent_id": agent_id,
            "submitted_epoch": time.time(),
            "queue_ahead": self.q.qsize(),   # jobs already waiting ahead of this one
            "done": threading.Event(),
            "result": None,
        }
        self.q.put(job)
        job["done"].wait()
        return job["result"]


class Handler(BaseHTTPRequestHandler):
    oracle = None  # assigned before serve_forever()

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            o = self.oracle
            self._send(200, {"ok": True, "n_problems": len(o.gold),
                             "index_min": min(o.gold), "index_max": max(o.gold),
                             "cost_c": o.cost_c, "queries_so_far": o.seq, "dataset": o.dataset})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/verify":
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            self._send(400, {"error": "bad json body"})
            return
        if "index" not in req or "candidate" not in req:
            self._send(400, {"error": "body needs 'index' and 'candidate'"})
            return
        try:
            index = int(req["index"])
        except Exception:
            self._send(400, {"error": "index must be an integer"})
            return
        if index not in self.oracle.gold:
            self._send(400, {"error": f"unknown index {index}",
                             "valid_range": [min(self.oracle.gold), max(self.oracle.gold)]})
            return
        result = self.oracle.submit(index, str(req["candidate"]),
                                    req.get("agent_id"), req.get("query_id"))
        self._send(200, result)

    def log_message(self, *args):  # silence per-request stderr noise
        pass


def main():
    cfg_path = os.environ.get("GRADER_CONFIG", os.path.join(HERE, "config.yaml"))
    cfg = yaml.safe_load(open(cfg_path))
    _load_secrets()
    audit = cfg.get("audit_log", "logs/queries.jsonl")
    if not os.path.isabs(audit):
        audit = os.path.abspath(audit)  # relative audit paths follow the working directory
    os.makedirs(os.path.dirname(audit), exist_ok=True)

    print(f"loading dataset: {cfg['dataset']['source']} ({cfg['dataset']['format']}) ...",
          flush=True)
    gold = load_gold(cfg)
    print(f"loaded {len(gold)} problems "
          f"(index {min(gold)}..{max(gold)}); cost_c={cfg['cost_c']}s", flush=True)

    dataset = dict(cfg['dataset'])
    if dataset['format'] == 'jsonl':
        source = dataset['source']
        if not os.path.isabs(source):
            source = os.path.join(HERE, source)
        with open(source, 'rb') as f:
            dataset['sha256'] = hashlib.sha256(f.read()).hexdigest()
    Handler.oracle = Oracle(gold, cfg["cost_c"], audit, dataset)
    host, port = cfg.get("host", "127.0.0.1"), int(cfg.get("port", 8077))
    srv = ThreadingHTTPServer((host, port), Handler)
    print(f"grader listening on http://{host}:{port}  "
          f"(POST /verify, GET /health)  — Ctrl-C to stop", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", flush=True)
        srv.shutdown()


if __name__ == "__main__":
    main()
