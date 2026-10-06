"""Validated canonical v1 configuration; services and model calls start elsewhere."""

import argparse
import shutil
import sys
from pathlib import Path
from reasoning_speedrun.lib.datasets import add_dataset_args, default_benchmark_role
from reasoning_speedrun.lib.storage import add_benchmark_args, apply_benchmark_args


def default_vllm_binary():
    beside = Path(sys.executable).parent / "vllm"
    return str(beside if beside.exists() else shutil.which("vllm") or beside)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_dataset_args(parser)
    parser.add_argument(
        "--model",
        default="r0b0tlab/VibeThinker-3B-NVFP4",
        help="Model ID with a launch profile under ~/models",
    )
    parser.add_argument("--models-dir", default="~/models")
    parser.add_argument(
        "--vllm-python",
        default=sys.executable,
        help="Python with vLLM installed (default: the interpreter running this command)",
    )
    parser.add_argument(
        "--vllm-binary",
        default=default_vllm_binary(),
        help="vllm executable (default: beside the current interpreter, else on PATH)",
    )
    parser.add_argument(
        "--grader-python",
        default=sys.executable,
        help="Python with the grader extra installed (default: the current interpreter)",
    )
    parser.add_argument("--reuse-server", action="store_true")
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--grader-port", type=int, default=8077)
    parser.add_argument("--grader-cost", type=float, default=3.0)
    parser.add_argument("--schedule", choices=("eager", "barrier"), default="barrier")
    parser.add_argument("--model-profile", default="vllm-flashinfer.yaml")
    parser.add_argument(
        "--parallelism", type=int, default=30, help="Concurrent question groups"
    )
    parser.add_argument(
        "--rollouts",
        type=int,
        default=1,
        help="Concurrent samples per question group; initial pass uses 8K each",
    )
    parser.add_argument("--first-pass-max-tokens", type=int, default=8192)
    parser.add_argument(
        "--no-continuation",
        action="store_true",
        help="Coverage: use fresh samples even after capped outputs",
    )
    parser.add_argument("--target-correct", type=int, default=18)
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=4,
        help="Coverage round bound, including first pass",
    )
    parser.add_argument(
        "--max-attempts-per-question",
        type=int,
        default=4,
        help="Counts every generation request, including continuations",
    )
    parser.add_argument(
        "--questions",
        type=int,
        nargs="+",
        help="Smoke subset; default all dataset questions",
    )
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--disable-thinking", action="store_true")
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--question-timeout", type=float, default=1800)
    parser.add_argument("--gpu-interval", type=float, default=0.2)
    parser.add_argument("--gpu-device", type=int, default=0)
    parser.add_argument("--no-overhead-profile", action="store_true")
    parser.add_argument(
        "--overhead-interval",
        type=float,
        default=0.05,
        help="Event-loop lag sampling seconds",
    )
    parser.add_argument(
        "--engine-metrics-interval",
        type=float,
        default=1.0,
        help="vLLM metrics polling seconds",
    )
    prewarming = parser.add_mutually_exclusive_group()
    prewarming.add_argument(
        "--benchmark-prewarm",
        dest="skip_benchmark_prewarm",
        action="store_false",
        help="Opt in to the ungraded 30-question AIME 2024 workload",
    )
    prewarming.add_argument(
        "--skip-benchmark-prewarm",
        action="store_true",
        help="Use only the inexpensive short warmup (default)",
    )
    parser.set_defaults(skip_benchmark_prewarm=True)
    parser.add_argument("--prewarm-max-tokens", type=int, default=8192)
    add_benchmark_args(parser)
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable CPU/engine/GPU profiling; retain buffered trace storage",
    )
    args = parser.parse_args(argv)
    if args.profile:
        args.benchmark = False
        args.no_overhead_profile = args.no_gpu_telemetry = False
        args.buffer_traces = True
    args = apply_benchmark_args(args)
    args.benchmark_year = args.benchmark_year or (
        None
        if args.dataset_manifest or args.grader_config or args.reuse_grader
        else 2025
    )
    args.benchmark_role = args.benchmark_role or default_benchmark_role(
        args.benchmark_year
    )
    args.strategy = "coverage"
    if args.seed < 0 or (
        args.benchmark_year == 2024 and (not args.skip_benchmark_prewarm)
    ):
        parser.error(
            "Use a nonnegative seed and a test year different from the 2024 warmup"
        )
    if args.rollouts > args.max_attempts_per_question:
        parser.error("Rollouts exceed the per-question attempt limit")
    if args.max_attempts_per_question > 4:
        parser.error(
            "Core v1 has a hard ceiling of four generation requests per question"
        )
    if Path(
        args.model_profile
    ).name != args.model_profile or not args.model_profile.endswith(".yaml"):
        parser.error("Model profile must be a YAML filename within the model directory")
    if any(
        (
            getattr(args, key) <= 0
            for key in (
                "prewarm_max_tokens",
                "parallelism",
                "rollouts",
                "max_tokens",
                "startup_timeout",
                "question_timeout",
                "gpu_interval",
                "max_rounds",
                "first_pass_max_tokens",
                "target_correct",
                "max_attempts_per_question",
                "overhead_interval",
                "engine_metrics_interval",
            )
        )
    ):
        parser.error(
            "Concurrency, token budgets, timeouts, and sampling interval must be positive"
        )
    if (
        args.grader_cost < 0
        or args.gpu_device < 0
        or (not 0 < args.top_p <= 1)
        or (args.temperature < 0)
    ):
        parser.error("Invalid grader cost, GPU device, or sampling settings")
    if (
        "/" not in args.model
        or any((part in ("", ".", "..") for part in args.model.split("/")))
        or args.model.startswith("/")
    ):
        parser.error("Use a relative organization/model ID")
    if (
        not all((1 <= port <= 65535 for port in (args.vllm_port, args.grader_port)))
        or args.vllm_port == args.grader_port
    ):
        parser.error("Service ports must be valid and distinct")
    args.max_context_tokens = 32768
    args.vllm_url = f"http://127.0.0.1:{args.vllm_port}"
    args.grader_url = f"http://127.0.0.1:{args.grader_port}"
    return args
