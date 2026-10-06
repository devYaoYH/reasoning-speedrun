"""Attempt lifecycle. The v1 policy lives in scheduler.py and question.py."""

import asyncio
from datetime import datetime, timezone
import hashlib
import subprocess
import sys
import time

import httpx

from reasoning_speedrun.integrity import verify_core
from reasoning_speedrun.lib.common import PACKAGE, atomic_json, utc_now, workdir
from reasoning_speedrun.lib.gpu import GPUSampler
from reasoning_speedrun.lib.metrics import AttemptProfiler
from reasoning_speedrun.lib.services import Services, ensure_free
from reasoning_speedrun.lib.setup import (
    local_dataset,
    prepare_grader,
    prepare_inference,
    read_profile,
)
from reasoning_speedrun.lib.storage import AttemptArtifacts, DisabledGPUSampler
from reasoning_speedrun.lib.summary import finalize
from reasoning_speedrun.lib.warmup import warm_inference
from reasoning_speedrun.scheduler import run_speedrun

RUNNER_ID = "runner_core_v1"


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


async def run(args):
    args.core_manifest_sha256 = verify_core(PACKAGE)
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
    config = {
        **vars(args),
        **git_state(),
        "attempt_id": output.name,
        "initialization_started_at_utc": utc_now(),
        "runner_id": RUNNER_ID,
        "runner_module": "reasoning_speedrun",
        "strategy": "coverage",
        "policy_origin": "runner_final_core_v1",
        "grading": "single vendored grader; no local answer-key comparisons",
        "gpu_scope": (
            "disabled for benchmark"
            if args.no_gpu_telemetry
            else "device-level NVML; vLLM preallocates VRAM"
        ),
        "python": sys.version,
    }
    atomic_json(output / "config.json", config)
    try:
        problems = local_dataset(args, config)
        profile_path, profile_text, profile = read_profile(args, output)
        if not args.reuse_grader:
            ensure_free(args.grader_port)
        if not args.reuse_server:
            ensure_free(args.vllm_port)
        sampler = (
            DisabledGPUSampler()
            if args.no_gpu_telemetry
            else GPUSampler(
                artifacts.gpu_path(output / "gpu.jsonl"),
                args.gpu_interval,
                args.gpu_device,
            )
        )
        await sampler.start()
        connections = args.parallelism * (args.rollouts + 1) + 8
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=10, read=args.question_timeout, write=30, pool=30
            ),
            limits=httpx.Limits(
                max_connections=connections, max_keepalive_connections=connections
            ),
            trust_env=False,
        ) as client:
            await prepare_inference(
                args, client, services, profiler, output, config, profile_path, profile
            )
            problems = await prepare_grader(
                args, client, services, profiler, output, config, problems
            )
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
            results = await run_speedrun(
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
        # Preserve the measured clock boundary: cleanup and buffered flush follow it.
        official_end = time.perf_counter()
        await finalize(
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
        )
    if status == "failed":
        raise RuntimeError("One or more questions failed; inspect trace telemetry")
    return output
