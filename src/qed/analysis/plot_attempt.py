"""Plot a saved attempt: requests in flight, verified-correct curve, request budgets.

    python -m qed.analysis.plot_attempt ATTEMPT_DIR OUTPUT_DIR

Reads only files the runner always saves (``config.json``, ``solved.jsonl``,
``trace/*/rollout-*/telemetry.json``). Requires the ``analysis`` extra
(``pip install -e '.[analysis]'``). Client request lifetimes include HTTP queueing and
cancellation time; they are not sampled engine-running counts.
"""

import argparse
from collections import Counter
from datetime import datetime
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def stamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def load(attempt):
    config = json.loads((attempt / "config.json").read_text())
    records = [
        json.loads(p.read_text())
        for p in attempt.glob("trace/*/rollout-*/telemetry.json")
    ]
    records = [r for r in records if r.get("generation_end_monotonic_s") is not None]
    if not records:
        raise ValueError("No completed rollout telemetry in this attempt")
    solved = sorted(
        json.loads(line)["first_solved_elapsed_s"]
        for line in (attempt / "solved.jsonl").read_text().splitlines()
        if line.strip()
    )
    return config, records, solved


def timeline(config, records):
    """Step series of fresh and continuation requests in flight, in official seconds."""
    earliest = min(records, key=lambda r: r["start_monotonic_s"])
    origin = earliest["start_monotonic_s"] - (
        stamp(earliest["started_at_utc"]) - stamp(config["official_started_at_utc"])
    )
    events = []
    for r in records:
        lane = 0 if r.get("continuation_of_rollout") is None else 1
        events += [
            (r["start_monotonic_s"] - origin, 1, lane),
            (r["generation_end_monotonic_s"] - origin, -1, lane),
        ]
    times, fresh, continued, active = [0.0], [0], [0], [0, 0]
    # Ends sort before starts at equal times (delta -1 < +1).
    for t, delta, lane in sorted(events):
        active[lane] += delta
        times.append(t)
        fresh.append(active[0])
        continued.append(active[1])
    return times, fresh, continued


def render(attempt, output):
    config, records, solved = load(attempt)
    times, fresh, continued = timeline(config, records)
    target = config.get("target_correct")
    budgets = Counter(r.get("requested_max_tokens") for r in records)
    fig, axes = plt.subplots(
        3, 1, figsize=(10, 8), layout="constrained",
        gridspec_kw={"height_ratios": [2, 1.2, 1.2]},
    )
    axes[0].stackplot(
        times, fresh, continued, step="post",
        labels=["Fresh samples", "Continuation segments"],
        colors=["#2588ad", "#dc8d39"], alpha=0.8,
    )
    axes[0].set_ylabel("Client requests in flight")
    axes[0].set_xlabel("Official elapsed seconds")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[1].step([0, *solved], range(len(solved) + 1), where="post", color="#268662", lw=2)
    axes[1].set_ylabel("Verified correct")
    axes[1].set_xlabel("Official elapsed seconds")
    if target:
        axes[1].axhline(target, color="#485664", ls="--", lw=1)
        axes[1].set_ylim(0, target * 1.15)
        if len(solved) >= target:
            axes[1].text(solved[target - 1], target + 0.4, f"{target} at {solved[target - 1]:.2f}s", ha="right")
    keys = sorted(budgets, key=lambda k: (k is None, k))
    axes[2].bar([f"{k // 1024}K" if k else "n/a" for k in keys], [budgets[k] for k in keys], color="#4d7897")
    axes[2].set_xlabel("Requested max tokens per request")
    axes[2].set_ylabel("Generation requests")
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle(f"{attempt.name} · {config.get('model', 'unknown model')}", fontsize=12)
    output.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(output / f"attempt-timeline.{ext}", dpi=170)
    plt.close(fig)
    result = {
        "peak_client_requests_in_flight": max(map(sum, zip(fresh, continued))),
        "fresh_requests": sum(r.get("continuation_of_rollout") is None for r in records),
        "continuation_requests": sum(r.get("continuation_of_rollout") is not None for r in records),
        "requests_by_requested_max_tokens": {str(k): v for k, v in budgets.items()},
        "verified_correct": len(solved),
        "time_to_target_s": solved[target - 1] if target and len(solved) >= target else None,
    }
    (output / "attempt-timeline.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("attempt", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    print(json.dumps(render(args.attempt, args.output), indent=2))


if __name__ == "__main__":
    main()
