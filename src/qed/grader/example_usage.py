"""Mock example: how another Python program queries the speedrun grader.

1) Start the server in one terminal:   $PY server.py
2) Run this in another:                $PY example_usage.py

Uses only the stdlib (urllib) so it's obvious how to call from ANY program /
language: it's just HTTP POST of {"index", "candidate"} to /verify.
"""
import json
import time
import threading
import urllib.request
import urllib.error

BASE = "http://127.0.0.1:8077"


def verify(index, candidate, agent_id=None, base=BASE):
    """Ask the grader whether `candidate` is correct for problem `index`.

    Returns the result dict (incl. verdict + answered_at time-tag). BLOCKS until
    the grader answers — under load it may wait, because the grader answers at most
    once every cost_c seconds globally. So use a long read timeout.
    """
    payload = json.dumps({"index": index, "candidate": candidate,
                          "agent_id": agent_id}).encode()
    req = urllib.request.Request(base + "/verify", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())


def health(base=BASE):
    with urllib.request.urlopen(base + "/health", timeout=10) as r:
        return json.loads(r.read())


if __name__ == "__main__":
    info = health()
    print("grader:", info)
    c = info["cost_c"]

    # --- 1. a single verification ---
    # Every verification costs cost_c (the grader "takes time"), so even this first
    # one waits ~cost_c before answering.
    t0 = time.time()
    res = verify(info["index_min"], "204")   # problem, candidate answer
    print(f"\n[single] verdict={res['verdict']} seq={res['seq']} "
          f"query_id={res['query_id'][:8]} answered_at={res['answered_at']} "
          f"(waited {time.time()-t0:.1f}s)")

    # --- 2. many CONCURRENT verifications from many threads ---
    # They all fire at once, but the grader serializes them: each costs cost_c, so
    # answers come back one every cost_c seconds, in arrival order, each time-tagged.
    # This is how a batched solver submits candidates — the gate, not the client, paces.
    idxs = list(range(info["index_min"], info["index_min"] + 5))
    print(f"\n[concurrent] firing {len(idxs)} queries at once "
          f"(expect ~{len(idxs)*c:.0f}s total, one answer per {c:.0f}s)...")
    results = {}

    def worker(i):
        results[i] = verify(i, str(i), agent_id=f"agent-{i}")

    t0 = time.time()
    threads = [threading.Thread(target=worker, args=(i,)) for i in idxs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for i in sorted(results, key=lambda k: results[k]["seq"]):
        r = results[i]
        print(f"  seq={r['seq']} q={r['query_id'][:8]} index={r['index']} "
              f"verdict={r['verdict']} ahead={r['queue_ahead_at_submit']} "
              f"answered_at={r['answered_at']} latency={r['latency_s']}s")
    print(f"[concurrent] total wall-time: {time.time()-t0:.1f}s")
