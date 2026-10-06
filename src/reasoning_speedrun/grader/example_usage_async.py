"""Mock example #2 — ASYNC client firing a MIX of correct and incorrect answers.

Start the server first (defaults to the bundled AIME-2025 file, no network needed):
    .venv/bin/python server.py
Then, in another shell:
    .venv/bin/python example_usage_async.py

Demonstrates:
  * querying the grader concurrently from asyncio (the blocking HTTP call is run in a
    thread executor — the standard stdlib way to go async without extra deps),
  * a mix of correct and deliberately-wrong candidates, checking each verdict,
  * a printed client-side input/output log, with wall-clock + per-query time-tags.

NOTE: this demo reads the local dataset's gold answers ONLY to decide which candidates
to send (correct vs wrong) and what to expect back. A real speedrun solver would NOT
have the gold — it would submit its own attempts and learn yes/no from the grader.
"""
import os
import sys
import json
import time
import asyncio
import urllib.request
import urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "http://127.0.0.1:8077"


def _verify(index, candidate, agent_id=None, query_id=None, base=BASE):
    """Blocking HTTP POST /verify -> result dict (run inside an executor for async)."""
    payload = json.dumps({"index": index, "candidate": candidate,
                          "agent_id": agent_id, "query_id": query_id}).encode()
    req = urllib.request.Request(base + "/verify", data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())


def _health(base=BASE):
    with urllib.request.urlopen(base + "/health", timeout=10) as r:
        return json.loads(r.read())


def load_gold():
    """Read {index: gold} from the dataset named in config.yaml (local jsonl)."""
    import yaml
    cfg = yaml.safe_load(open(os.path.join(HERE, "config.yaml")))
    d = cfg["dataset"]
    if d["format"] != "jsonl":
        raise SystemExit("this demo expects a local jsonl dataset (the default config).")
    path = d["source"]
    if not os.path.isabs(path):
        path = os.path.join(HERE, path)
    gold = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            gold[int(r[d["idx_field"]])] = str(r[d["gold_field"]])
    return gold


def _clock():
    return time.strftime("%H:%M:%S")


async def main():
    try:
        info = _health()
    except urllib.error.URLError:
        sys.exit("could not reach the grader — start it first:  .venv/bin/python server.py")
    c = info["cost_c"]
    print(f"grader: {info}\n")
    gold = load_gold()

    # Build a MIX: even positions correct (candidate == gold), odd positions wrong.
    plan = []   # (index, candidate, expected_verdict)
    for n, i in enumerate(sorted(gold)[:6]):
        if n % 2 == 0:
            plan.append((i, gold[i], True))                 # correct
        else:
            try:
                wrong = str(int(gold[i]) + 1)               # numeric -> +1, always != gold
            except ValueError:
                wrong = gold[i] + "_x"                       # non-numeric -> guaranteed different
            plan.append((i, wrong, False))                  # deliberately wrong

    loop = asyncio.get_running_loop()

    async def ask(index, candidate, expected):
        qid = f"demo-{index}"
        print(f"  -> {_clock()} SUBMIT  idx={index:>2}  candidate={candidate!r:>6}  "
              f"expect={expected!s:<5} query_id={qid}")
        res = await loop.run_in_executor(None, _verify, index, candidate, "async-demo", qid)
        match = (res["verdict"] == expected)
        print(f"  <- {_clock()} {'OK ' if match else '!! '} seq={res['seq']:>2} "
              f"q={res['query_id']:<8} idx={res['index']:>2} verdict={res['verdict']!s:<5} "
              f"(expected {expected!s:<5}) latency={res['latency_s']:>5}s")
        return res, expected, match

    print(f"[async] firing {len(plan)} queries concurrently — a mix of correct/incorrect "
          f"(~{len(plan)*c:.0f}s total, one answer per {c:.0f}s):\n")
    t0 = time.time()
    results = await asyncio.gather(*(ask(i, cand, exp) for i, cand, exp in plan))
    total = time.time() - t0

    matched = sum(1 for _, _, m in results if m)
    print(f"\n[async] {matched}/{len(plan)} verdicts matched expectation; "
          f"total wall-time {total:.1f}s. Full server-side record -> logs/queries.jsonl")
    if matched != len(plan):
        sys.exit("MISMATCH: a verdict did not match expectation")


if __name__ == "__main__":
    asyncio.run(main())
