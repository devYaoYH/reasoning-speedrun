"""Official timings, durable summaries and post-clock cleanup/trace flush."""

import json
from pathlib import Path
import time
from qed.lib.common import atomic_json, utc_now
from qed.lib.metadata import build_metadata
from qed.lib.metrics import grader_timeline

RUNNER_ID = "runner_core_v1"


def ttft(rollouts, continued):
    values = sorted(
        (
            r["ttft_s"]
            for r in rollouts
            if r["ttft_s"] is not None
            and (r["continuation_of_rollout"] is not None) == continued
        )
    )
    return {
        "count": len(values),
        "median_s": values[len(values) // 2] if values else None,
        "p95_s": values[int((len(values) - 1) * 0.95)] if values else None,
    }


async def finalize(
    args,
    output,
    config,
    artifacts,
    profiler,
    sampler,
    services,
    status,
    error,
    official_start,
    official_end,
    began,
    results,
):
    summary = {
        "attempt_id": output.name,
        "status": status,
        "error": error,
        "official_started_at_utc": config.get("official_started_at_utc"),
        "official_finished_at_utc": utc_now(),
        "official_latency_s": official_end - official_start if official_start else None,
        "initialization_and_attempt_latency_s": official_end - began,
        "benchmark_prewarm": config.get("benchmark_prewarm"),
        "runner_id": RUNNER_ID,
        "schedule": args.schedule,
        "solved": sum((r["status"] == "solved" for r in results)),
        "questions_completed": sum((r["status"] != "stopped" for r in results)),
        "questions_attempted": len(results),
        "target_correct": args.target_correct,
        "target_reached": sum((r["status"] == "solved" for r in results))
        >= args.target_correct,
        "rounds_executed": max((r["round"] for r in results), default=0),
        "questions": [
            {
                **{
                    k: r[k]
                    for k in (
                        "problem_idx",
                        "status",
                        "end_to_end_latency_s",
                        "unique_candidates",
                    )
                },
                "verified_answer": r["winner"]["candidate"] if r["winner"] else None,
                "winning_rollout": r["winner"]["rollout"] if r["winner"] else None,
                "first_solved": r.get("first_solved"),
            }
            for r in results
        ],
    }
    solved_times = sorted(
        (
            r["first_solved"]["first_solved_elapsed_s"]
            for r in results
            if r.get("first_solved")
        )
    )
    summary["time_to_target_s"] = (
        solved_times[args.target_correct - 1]
        if len(solved_times) >= args.target_correct
        else None
    )
    rollouts = [rollout for question in results for rollout in question["rollouts"]]
    summary["performance"] = {
        "generation_requests": len(rollouts),
        "fresh_ttft": ttft(rollouts, False),
        "continuation_ttft": ttft(rollouts, True),
        "official_gpu": (
            sampler.window(official_start, official_end)
            if sampler and official_start
            else None
        ),
        "initialization_gpu": (
            sampler.window(began, official_start or official_end) if sampler else None
        ),
    }
    summary["overhead"] = profiler.snapshot()
    atomic_json(output / "summary.json", summary)
    atomic_json(output / "config.json", config)
    cleanup_start = time.perf_counter()
    try:
        await services.close()
    finally:
        summary["service_cleanup_latency_s"] = time.perf_counter() - cleanup_start
        try:
            summary["trace_storage"] = artifacts.flush()
        except Exception as exc:
            summary.update(
                status="failed",
                error=f"Trace flush failed: {type(exc).__name__}: {exc}",
            )
            atomic_json(output / "summary.json", summary)
            raise
        summary["trace_storage"][
            "scope"
        ] = "Client trace flush after official timing; excluded from time to target and official latency"
        atomic_json(output / "summary.json", summary)
        try:
            atomic_json(output / "metadata.json", build_metadata(output, config))
        except Exception as exc:
            # A run that died before recording its dataset has nothing to normalize; its
            # own error is what matters, so never let metadata hide it.
            if summary.get("status") == "completed":
                raise
            summary["metadata_error"] = f"{type(exc).__name__}: {exc}"
    summary["grader_timeline"] = grader_timeline(
        output / "grader_audit.jsonl",
        config.get("official_started_at_utc"),
        args.target_correct,
        args.grader_cost,
    )
    atomic_json(output / "overhead.json", summary["overhead"])
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if official_start is not None:  # a run that never started has nothing to headline
        print(headline(summary, output, bool(config.get("simulated"))), flush=True)
    return summary


def headline(summary, output, simulated=False):
    """The one line a person wants after a run: did it reach the target, and how fast."""
    target, solved = summary["target_correct"], summary["solved"]
    try:
        where = output.relative_to(Path.cwd())
    except ValueError:
        where = output
    if summary.get("target_reached") and summary.get("time_to_target_s") is not None:
        result = f"reached {target} correct in {summary['time_to_target_s']:.1f} s"
    elif summary.get("status") == "completed":
        result = f"target not reached: {solved} of {target} correct when the budget ran out"
    else:
        result = f"run {summary.get('status')}: {solved} of {target} correct"
    note = " (simulated backend: not a hardware measurement)" if simulated else ""
    return f"\nqed: {result}{note}\n     saved to {where}; browse it with `qed view`"
