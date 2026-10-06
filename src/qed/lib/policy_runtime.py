"""Extension lifecycle; canonical v1 remains pinned to its original runtime."""

import asyncio
from datetime import datetime, timezone
import hashlib
import json
import subprocess
import sys
import time

import httpx

from qed.lib.common import PACKAGE, atomic_json, utc_now, workdir
from qed.lib.summary import ttft
from qed.lib.metrics import grader_timeline
from qed.lib.gpu import GPUSampler
from qed.lib.metrics import AttemptProfiler
from qed.lib.services import Services, ensure_free
from qed.lib.setup import (
    local_dataset,
    prepare_grader,
    prepare_inference,
    read_profile,
)
from qed.lib.storage import AttemptArtifacts, DisabledGPUSampler
from qed.sim import Simulation
from qed.lib.warmup import warm_inference


def git_state():
    """Installed packages usually run outside Git; the source manifest still pins them."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PACKAGE, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                cwd=PACKAGE,
                text=True,
            )
        )
        return {"git_commit": commit, "git_dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"git_commit": None, "git_dirty": None}


async def run_policy(
    args,
    *,
    policy,
    prepare_problems,
    verify,
    runner_id,
    runner_module,
    metadata_builder,
):
    args.core_manifest_sha256 = verify(PACKAGE)
    if not isinstance(args.system_prompt, str) or not args.system_prompt.strip():
        raise ValueError("A nonempty system prompt must be supplied by the entrypoint")
    services, sampler = Services(), None
    began = time.perf_counter()
    output = (
        workdir() / "attempts" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    )
    output.mkdir(parents=True)
    print(f"Attempt artifacts: {output}", flush=True)
    artifacts = AttemptArtifacts(output, buffered=args.buffer_traces)
    profiler = AttemptProfiler(
        output,
        artifacts=artifacts,
        enabled=not args.no_overhead_profile,
        interval=args.overhead_interval,
        engine_interval=args.engine_metrics_interval,
    )
    status, error, official_start = "initializing", None, None
    simulation = None
    config = {
        **vars(args),
        **git_state(),
        "attempt_id": output.name,
        "initialization_started_at_utc": utc_now(),
        "runner_id": runner_id,
        "runner_module": runner_module,
        "strategy": "coverage",
        "policy_origin": "v1 streaming extraction with extension scheduling and budgets",
        "grading": "single vendored grader; no local answer-key comparisons",
        "gpu_scope": (
            "disabled for benchmark"
            if args.no_gpu_telemetry
            else (
                "simulated device; no GPU"
                if args.simulate
                else "device-level NVML; vLLM preallocates VRAM"
            )
        ),
        "python": sys.version,
    }
    atomic_json(output / "config.json", config)
    try:
        problems = local_dataset(args, config)
        profile_path, profile_text, profile = read_profile(args, output)
        simulation = Simulation.create(args, config, profile) if args.simulate else None
        if not args.reuse_grader:
            ensure_free(args.grader_port)
        if not (args.reuse_server or args.simulate):
            ensure_free(args.vllm_port)
        gpu_path = artifacts.gpu_path(output / "gpu.jsonl")
        sampler = (
            DisabledGPUSampler()
            if args.no_gpu_telemetry
            else (
                simulation.gpu_sampler(gpu_path, args.gpu_interval)
                if simulation
                else GPUSampler(gpu_path, args.gpu_interval, args.gpu_device)
            )
        )
        await sampler.start()
        connections = args.parallelism * (args.rollouts + 1) + 8
        limits = httpx.Limits(
            max_connections=None, max_keepalive_connections=connections
        )
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10, read=args.question_timeout, write=30, pool=30
            ),
            limits=limits,
            trust_env=False,
            **(simulation.client_kwargs(limits) if simulation else {}),
        ) as client:
            await prepare_inference(
                args, client, services, profiler, output, config, profile_path, profile
            )
            problems = await prepare_grader(
                args, client, services, profiler, output, config, problems
            )
            problems = await prepare_problems(problems, args, client)
            if (
                config["dataset_provenance"].get("year") == 2024
                and not args.skip_benchmark_prewarm
            ):
                raise ValueError("Test dataset must differ from the AIME 2024 warmup")
            with profiler.meter.measure(
                "initialization_inference_warmup_wait", cpu=False
            ):
                warmup = await warm_inference(args, client, len(problems))
            atomic_json(output / "inference_warmup.json", warmup)
            config["inference_warmup"] = {
                k: warmup[k] for k in ("latency_s", "batch_size", "tokens_per_request")
            }
            config.update(vars(args))
            config.update(
                model_profile_sha256=hashlib.sha256(profile_text.encode()).hexdigest(),
                launch_profile=profile,
                official_started_at_utc=utc_now(),
            )
            atomic_json(output / "config.json", config)
            official_start = time.perf_counter()
            profiler.official_start(client, args.vllm_url)
            status = "running"
            print("Official solving phase started", flush=True)
            results = await policy(
                problems, args, client, output, sampler, official_start, profiler
            )
            await profiler.stop()
            status = (
                "completed"
                if all(r["status"] != "error" for r in results)
                else "failed"
            )
            if sampler.error:
                raise RuntimeError(f"GPU telemetry failed: {sampler.error}")
    except asyncio.CancelledError:
        status, error = "interrupted", "Attempt interrupted by signal"
        raise
    except Exception as exc:
        status, error = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        await profiler.stop()
        if sampler:
            sampler.stop()
        if simulation:
            atomic_json(output / "simulation.json", simulation.report())
        # Preserve the measured clock boundary: cleanup and buffered flush follow it.
        official_end = time.perf_counter()
        await finalize_policy(
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
            artifacts.questions(),
            runner_id=runner_id,
            metadata_builder=metadata_builder,
        )
    if status == "failed":
        raise RuntimeError("One or more questions failed; inspect trace telemetry")
    return output


async def finalize_policy(
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
    *,
    runner_id,
    metadata_builder,
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
        "runner_id": runner_id,
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
        "fresh_samples": sum(r["continuation_of_rollout"] is None for r in rollouts),
        "continuation_requests": sum(
            r["continuation_of_rollout"] is not None for r in rollouts
        ),
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
        atomic_json(output / "metadata.json", metadata_builder(output, config))
    summary["grader_timeline"] = grader_timeline(
        output / "grader_audit.jsonl",
        config.get("official_started_at_utc"),
        args.target_correct,
        args.grader_cost,
    )
    atomic_json(output / "overhead.json", summary["overhead"])
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary
