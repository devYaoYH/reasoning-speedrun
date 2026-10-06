"""Deterministic replay of a saved attempt under different scheduling assumptions.

    python -m reasoning_speedrun.simulate ATTEMPT_DIR --compare
    python -m reasoning_speedrun.simulate ATTEMPT_DIR --slots 30 --cost 3 --extraction final

The replay takes the *recorded* trajectories (start, duration, the candidate
answers they emitted and when) and the *recorded* grader verdicts, then re-runs
the policy-dependent machinery: slot-bounded admission, FIFO verification at a
chosen cost, deduplication before grading, early exit on a verified answer, and
final-answer-only extraction. No model or grader is contacted and no gold answer
is read: correctness comes only from verdicts the original run obtained.

What it cannot know (reported, never guessed):

* Durations are replayed as recorded. Decode slowing down as concurrency grows
  is not modelled, so concurrency/``stream_seconds`` figures are an occupancy
  proxy for GPU contention, not a throughput prediction.
* Streams cancelled after their question was solved are *censored*: their true
  end and any later candidates were never observed.  ``censored="drop"`` treats
  their final answers as never arriving, ``"lower_bound"`` as arriving at the
  moment they were cancelled; with final-only extraction these bracket the truth.
* A candidate never sent to the grader has no verdict and is skipped (counted as
  ``unresolved``). Duplicate answers were suppressed in the recording, so a
  trajectory's candidate list can be incomplete.

The recorded policy replayed with ``slots=None`` should reproduce the recorded
time to target; ``--compare`` prints that check first.
"""

import argparse
from collections import deque
from datetime import datetime
import heapq
import itertools
import json
from pathlib import Path


def stamp(value):
    return datetime.fromisoformat(value).timestamp()


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def read_lines(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_trajectories(attempt):
    """Rebuild fresh-sample trajectories (continuation segments chained) from a saved attempt."""
    attempt = Path(attempt)
    config = read_json(attempt / "config.json")
    if not config or not config.get("official_started_at_utc"):
        raise ValueError(f"{attempt}: config.json lacks an official start time")
    origin = stamp(config["official_started_at_utc"])
    trajectories, verdicts = [], {}
    for folder in sorted(attempt.glob("trace/[0-9]*")):
        question = read_json(folder / "question.json")
        if not question:
            continue
        index = question["problem_idx"]
        events = read_lines(folder / "verification.jsonl")
        for event in events:
            verdict = (event.get("result") or {}).get("verdict")
            if isinstance(verdict, bool):
                verdicts[(index, event["candidate"])] = verdict
        segments = {}
        for r in question["rollouts"]:
            if not r.get("generation_finished_at_utc"):
                continue
            segments[r["rollout"]] = {
                "rollout": r["rollout"],
                "parent": r.get("continuation_of_rollout"),
                "start": stamp(r["started_at_utc"]) - origin,
                "end": stamp(r["generation_finished_at_utc"]) - origin,
                "censored": bool(r.get("generation_censored")),
                "finish_reason": r.get("finish_reason"),
                "tokens": (r.get("usage") or {}).get("completion_tokens"),
            }
        chains = {}
        for seg in segments.values():
            root = seg
            while root["parent"] in segments:
                root = segments[root["parent"]]
            chains.setdefault(root["rollout"], []).append(seg)
        for root, chain in chains.items():
            chain.sort(key=lambda s: s["start"])
            active = 0.0
            for seg in chain:
                seg["active_before"] = active
                active += seg["end"] - seg["start"]
            trajectories.append(
                {
                    "question": index,
                    "id": f"q{index}-s{root}",
                    "segments": chain,
                    "start": chain[0]["start"],
                    "end": chain[-1]["end"],
                    "active": active,
                    # The trajectory's own end was observed only if its last segment
                    # ended on its own (stop), not by a cap or cancellation.
                    "censored": chain[-1]["censored"] or chain[-1]["finish_reason"] != "stop",
                    "candidates": [],
                }
            )
        by_rollout = {}
        for t in trajectories:
            if t["question"] == index:
                for seg in t["segments"]:
                    by_rollout[seg["rollout"]] = (t, seg)
        for event in events:
            owner = by_rollout.get(event["rollout"])
            if owner is None:
                continue
            t, seg = owner
            arrival = stamp(event["observed_at_utc"]) - origin
            offset = seg["active_before"] + min(max(arrival - seg["start"], 0.0), seg["end"] - seg["start"])
            t["candidates"].append(
                {
                    "answer": event["candidate"],
                    "kind": event.get("kind"),
                    "arrival": arrival,
                    "offset": offset,
                    "verdict": verdicts.get((index, event["candidate"])),
                }
            )
    for t in trajectories:
        t["candidates"].sort(key=lambda c: c["offset"])
    trajectories.sort(key=lambda t: (t["start"], t["question"], t["id"]))
    recorded = sorted(
        (r["first_solved_elapsed_s"] for r in read_lines(attempt / "solved.jsonl")
         if r.get("first_solved_elapsed_s") is not None)
    )
    return {
        "trajectories": trajectories,
        "verdicts": verdicts,
        "config": config,
        "recorded_solved_s": recorded,
    }


def candidate_view(trajectory, extraction, censored):
    """Candidates a policy would see: (arrival offset in active time, answer, verdict, kind)."""
    if extraction == "early":
        return [(c["offset"], c) for c in trajectory["candidates"]]
    if not trajectory["candidates"]:
        return []
    if trajectory["censored"] and censored == "drop":
        return []
    # Final-only: one answer per trajectory, available when it ends. Its answer is
    # the last candidate the trajectory emitted.
    return [(trajectory["active"], trajectory["candidates"][-1])]


def simulate(
    data,
    *,
    slots=None,
    cost_s=None,
    extraction="early",
    censored="drop",
    target=None,
    cancel_on_solve=True,
):
    """Replay ``data`` (from ``load_trajectories``); ``slots=None`` keeps recorded starts."""
    if extraction not in ("early", "final") or censored not in ("drop", "lower_bound"):
        raise ValueError("extraction is early|final and censored is drop|lower_bound")
    if slots is not None and slots < 1:
        raise ValueError("slots must be positive")
    config = data["config"]
    cost_s = config.get("grader_cost", 3.0) if cost_s is None else cost_s
    target = config.get("target_correct", 18) if target is None else target
    if cost_s <= 0 or target < 1:
        raise ValueError("cost and target must be positive")
    trajectories = data["trajectories"]
    recorded_mode = slots is None

    heap, serial = [], itertools.count()
    PRIORITY = {"verify_done": 0, "candidate": 1, "end": 2, "start": 3}

    def push(t, kind, payload):
        heapq.heappush(heap, (t, PRIORITY[kind], next(serial), kind, payload))

    def span(traj):
        # Recorded starts keep real wall time (barrier gaps included); admitted
        # trajectories occupy a slot only while generating.
        return traj["end"] - traj["start"] if recorded_mode else traj["active"]

    waiting = deque(trajectories)
    live = {}  # id -> {"traj", "start"}
    intervals = []  # (start, end, trajectory, cancelled)
    solved, seen, queue = {}, set(), deque()
    checks, state = [], {"busy": False, "unresolved": 0, "discarded": 0, "peak_queue": 0}
    time_to_target = [None]

    def begin(traj, t):
        live[traj["id"]] = {"traj": traj, "start": t}
        for offset, cand in candidate_view(traj, extraction, censored):
            if recorded_mode:
                arrival = cand["arrival"] if extraction == "early" else traj["end"]
            else:
                arrival = t + offset
            push(arrival, "candidate", (traj, cand))
        push(t + span(traj), "end", traj["id"])

    def admit(t):
        while waiting and len(live) < slots:
            traj = waiting.popleft()
            if cancel_on_solve and traj["question"] in solved:
                continue
            begin(traj, t)

    def close(tid, t, cancelled):
        entry = live.pop(tid, None)
        if entry:
            intervals.append((entry["start"], t, entry["traj"], cancelled))

    def start_check(t):
        if state["busy"]:
            return
        while queue:
            job = queue.popleft()
            if job["q"] in solved:
                state["discarded"] += 1
                continue
            state["busy"] = True
            push(t + cost_s, "verify_done", {**job, "started": t, "queue_s": t - job["arrival"]})
            return

    if recorded_mode:
        for traj in trajectories:
            push(traj["start"], "start", traj)
    else:
        admit(0.0)
    now = 0.0
    while heap:
        now, _, _, kind, payload = heapq.heappop(heap)
        if kind == "candidate":
            traj, cand = payload
            q = traj["question"]
            if traj["id"] not in live or (cancel_on_solve and q in solved):
                continue
            key = (q, cand["answer"])
            if key in seen:
                continue
            seen.add(key)
            if cand["verdict"] is None:
                state["unresolved"] += 1
                continue
            queue.append({"q": q, "answer": cand["answer"], "verdict": cand["verdict"],
                          "arrival": now, "trajectory": traj["id"], "kind": cand.get("kind")})
            state["peak_queue"] = max(state["peak_queue"], len(queue))
        elif kind == "start":
            if not (cancel_on_solve and payload["question"] in solved):
                begin(payload, now)
        elif kind == "end":
            close(payload, now, False)
            if not recorded_mode:
                admit(now)
        elif kind == "verify_done":
            state["busy"] = False
            checks.append({**payload, "verified_at": now})
            q = payload["q"]
            if payload["verdict"] and q not in solved:
                solved[q] = {"problem_idx": q, "time_s": now, "answer": payload["answer"],
                             "trajectory": payload["trajectory"]}
                if len(solved) >= target and time_to_target[0] is None:
                    time_to_target[0] = now
                    break
                if cancel_on_solve:
                    for tid in [i for i, e in live.items() if e["traj"]["question"] == q]:
                        close(tid, now, True)
                    if not recorded_mode:
                        admit(now)
        start_check(now)
    horizon = time_to_target[0] if time_to_target[0] is not None else now
    for tid in list(live):
        entry = live[tid]
        end = entry["start"] + span(entry["traj"])
        close(tid, min(end, horizon), end > horizon)

    def clipped(a, b):
        return max(0.0, min(b, horizon) - min(a, horizon))

    stream_seconds = sum(clipped(a, b) for a, b, _, _ in intervals)
    solved_by_horizon = {q for q, s in solved.items() if s["time_s"] <= horizon}
    steps = sorted(
        [(min(a, horizon), 1) for a, b, _, _ in intervals if a < horizon]
        + [(min(b, horizon), -1) for a, b, _, _ in intervals if a < horizon]
    )
    level = peak = 0
    series = []
    for t, d in sorted(steps, key=lambda s: (s[0], s[1])):
        level += d
        peak = max(peak, level)
        series.append((round(t, 6), level))
    milestones = sorted(solved.values(), key=lambda r: (r["time_s"], r["problem_idx"]))
    return {
        "slots": slots,
        "cost_s": cost_s,
        "extraction": extraction,
        "censored": censored if extraction == "final" else None,
        "cancel_on_solve": cancel_on_solve,
        "target": target,
        "solved": len(solved),
        "time_to_target_s": time_to_target[0],
        "checks": len(checks),
        "wrong_checks": sum(not c["verdict"] for c in checks),
        "unresolved_candidates": state["unresolved"],
        "discarded_queued_checks": state["discarded"],
        "peak_queued_checks": state["peak_queue"],
        "trajectories_started": len(intervals),
        "peak_concurrent_streams": peak,
        "stream_seconds": stream_seconds,
        "mean_concurrent_streams": stream_seconds / horizon if horizon else 0.0,
        "stream_seconds_on_unsolved_questions": sum(
            clipped(a, b) for a, b, t, _ in intervals if t["question"] not in solved_by_horizon
        ),
        "concurrency": series,
        "milestones": milestones,
    }


def scenarios(data):
    """The standard comparison: recorded reproduction, then counterfactual extraction modes."""
    unbounded = len(data["trajectories"]) or 1
    return [
        ("recorded policy (reproduction check)", dict()),
        ("early extraction, all recorded trajectories at once", dict(slots=unbounded, extraction="early")),
        ("final-only, censored streams dropped", dict(slots=unbounded, extraction="final", censored="drop")),
        ("final-only, censored streams at cancel time", dict(slots=unbounded, extraction="final", censored="lower_bound")),
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("attempt", type=Path)
    parser.add_argument("--slots", type=int, help="Bound concurrent streams (default: replay recorded starts)")
    parser.add_argument("--cost", type=float, dest="cost_s", help="Verifier seconds per check (default: recorded)")
    parser.add_argument("--extraction", choices=("early", "final"), default="early")
    parser.add_argument("--censored", choices=("drop", "lower_bound"), default="drop")
    parser.add_argument("--target", type=int)
    parser.add_argument("--no-cancel", action="store_true", help="Keep a question's streams after it is solved")
    parser.add_argument("--compare", action="store_true", help="Run the standard scenario table")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    data = load_trajectories(args.attempt)
    if args.compare:
        runs = [(name, simulate(data, **kw)) for name, kw in scenarios(data)]
    else:
        runs = [("custom", simulate(
            data, slots=args.slots, cost_s=args.cost_s, extraction=args.extraction,
            censored=args.censored, target=args.target, cancel_on_solve=not args.no_cancel))]
    recorded = data["recorded_solved_s"]
    target = runs[0][1]["target"]
    if args.json:
        print(json.dumps({"recorded_time_to_target_s": recorded[target - 1] if len(recorded) >= target else None,
                          "runs": [{"name": n, **{k: v for k, v in r.items() if k not in ("concurrency", "milestones")}} for n, r in runs]}, indent=2))
        return
    ref = f"{recorded[target - 1]:.3f}s" if len(recorded) >= target else "unmet"
    print(f"{args.attempt.name}: {len(data['trajectories'])} recorded trajectories; recorded time to {target}: {ref}")
    print(f"{'scenario':<50} {'to target':>10} {'checks':>7} {'wrong':>6} {'peak':>5} {'mean':>6} {'stream-s':>9} {'unres':>6}")
    for name, r in runs:
        t = f"{r['time_to_target_s']:.3f}s" if r["time_to_target_s"] is not None else f"unmet({r['solved']})"
        print(f"{name:<50} {t:>10} {r['checks']:>7} {r['wrong_checks']:>6} {r['peak_concurrent_streams']:>5} "
              f"{r['mean_concurrent_streams']:>6.1f} {r['stream_seconds']:>9.1f} {r['unresolved_candidates']:>6}")


if __name__ == "__main__":
    main()
