"""Model profile checks and service startup; no solving clock runs here."""

import asyncio
import hashlib
import json
import os
from pathlib import Path
import yaml
from qed.lib.common import PACKAGE, atomic_json
from qed.lib.datasets import (
    benchmark_paths,
    dataset_provenance,
    load_questions,
    manifest_path,
)
from qed.lib.grader_questions import fetch_questions
from qed.lib.gpu import assert_gpu_idle
from qed.lib.services import ready
from qed.lib.prewarm import prewarm_benchmark


def resolve_profile(args, output):
    """Prefer ``MODELS_DIR/MODEL/PROFILE``; else materialize the bundled profile.

    Bundled profiles name weights as ``~/models/<model>``. The materialized copy
    written to the attempt folder points at ``MODELS_DIR/MODEL`` instead.
    """
    models = Path(args.models_dir).expanduser()
    path = models / args.model / args.model_profile
    if path.is_file():
        return path
    bundled = PACKAGE / "profiles/models" / args.model / args.model_profile
    if not bundled.is_file():
        raise FileNotFoundError(
            f"No model profile at {path} and none bundled for {args.model}/{args.model_profile}"
        )
    profile = yaml.safe_load(bundled.read_text())
    profile["model"] = str(models / args.model)
    path = output / "vllm_profile.yaml"
    path.write_text(yaml.safe_dump(profile, sort_keys=False))
    return path


def read_profile(args, output):
    path = resolve_profile(args, output)
    raw = path.read_text()
    profile = yaml.safe_load(raw)
    if not isinstance(profile, dict):
        raise ValueError("Model profile must be a YAML mapping")
    overrides = profile.get("override-generation-config", {})
    if isinstance(overrides, str):
        overrides = json.loads(overrides)
    cap = overrides.get("max_new_tokens")
    requested = max(
        args.max_tokens,
        args.first_pass_max_tokens,
        0 if args.skip_benchmark_prewarm else args.prewarm_max_tokens,
    )
    if cap is not None and requested > cap:
        raise ValueError(f"Requested max_tokens exceeds model profile ceiling ({cap})")
    atomic_json(
        output / "model_profile.json",
        {
            "path": str(path),
            "yaml": raw,
            "sha256": hashlib.sha256(raw.encode()).hexdigest(),
        },
    )
    return path, raw, profile


def local_dataset(args, config):
    """Validate built-in AIME at startup only; custom questions come from the grader."""
    if args.dataset_manifest or args.grader_config or args.reuse_grader:
        return None
    config["dataset_provenance"] = dataset_provenance(
        args.benchmark_year, args.benchmark_role
    )
    problems = load_questions(args.questions, args.benchmark_year)
    if args.target_correct > len(problems):
        raise ValueError("Target correct exceeds the number of selected questions")
    config["question_indices"] = [p["problem_idx"] for p in problems]
    return problems


async def prepare_grader(args, client, services, profiler, output, config, problems):
    """Reuse only a fresh queue; never own or stop a reused service."""
    general = problems is None
    grader = None
    if not args.reuse_grader:
        if args.grader_config:
            grader_config = yaml.safe_load(
                manifest_path(args.grader_config).read_text()
            )
        else:
            grader_config = {
                "dataset": {
                    "source": str(
                        benchmark_paths(args.benchmark_year, args.dataset_manifest)[1]
                    ),
                    "format": "jsonl",
                    "idx_field": "problem_idx",
                    "gold_field": "answer",
                    "id": (
                        f"aime_{args.benchmark_year}"
                        if args.benchmark_year
                        else "configured_dataset"
                    ),
                    "year": args.benchmark_year,
                }
            }
            if args.dataset_manifest:
                grader_config["dataset"]["manifest"] = str(
                    manifest_path(args.dataset_manifest)
                )
            elif not general:
                grader_config["dataset"].update(
                    id=config["dataset_provenance"]["id"],
                    revision=config["dataset_provenance"]["revision"],
                )
        grader_config.update(
            cost_c=args.grader_cost,
            host="127.0.0.1",
            port=args.grader_port,
            audit_log=str(output / "grader_audit.jsonl"),
        )
        path = output / "grader_config.yaml"
        path.write_text(yaml.safe_dump(grader_config))
        script = "server_questions.py" if general else "server.py"
        grader = services.launch(
            [args.grader_python, str(PACKAGE / "grader" / script)],
            output / "grader.log",
            {**os.environ, "GRADER_CONFIG": str(path)},
        )
    with profiler.meter.measure("initialization_grader_ready_wait", cpu=False):
        health = await ready(client, args.grader_url + "/health", 60, grader)
    if health.get("queries_so_far") != 0 or health.get("cost_c") != args.grader_cost:
        raise RuntimeError("Grader did not start with a fresh queue and requested toll")
    if general:
        problems, all_questions, dataset, digest = await fetch_questions(
            client, args, health
        )
        atomic_json(output / "questions.json", all_questions)
        dataset.update(
            prompt_path="questions.json",
            prompt_sha256=hashlib.sha256(
                (output / "questions.json").read_bytes()
            ).hexdigest(),
        )
        config.update(
            dataset_provenance=dataset,
            grader_questions_sha256=digest,
            question_indices=[p["problem_idx"] for p in problems],
            question_source="grader GET /questions; gold-free solver inputs",
        )
    elif (
        health.get("dataset", {}).get("sha256")
        != config["dataset_provenance"]["grader_sha256"]
    ):
        raise RuntimeError("Grader loaded a different benchmark answer key")
    else:
        atomic_json(output / "questions.json", problems)
    config["grader_health"] = health
    return problems


async def prepare_inference(
    args, client, services, profiler, output, config, profile_path, profile
):
    if args.simulate:
        server = None  # the simulated backend answers in process; no GPU or vLLM
    elif not args.reuse_server:
        assert_gpu_idle(args.gpu_device)
        gpu_warmup = services.launch(
            [
                args.vllm_python,
                "-c",
                "import torch,time; torch.cuda.set_device("
                + str(args.gpu_device)
                + '); x=torch.randn((1024,1024),device="cuda",dtype=torch.bfloat16); end=time.monotonic()+2; \nwhile time.monotonic()<end: y=x@x; torch.cuda.synchronize()',
            ],
            output / "gpu_warmup.log",
        )
        with profiler.meter.measure("initialization_gpu_warmup_wait", cpu=False):
            await asyncio.wait_for(asyncio.to_thread(gpu_warmup.wait), timeout=120)
        if gpu_warmup.returncode:
            raise RuntimeError("CUDA warmup failed; inspect gpu_warmup.log")
        command = [
            args.vllm_binary,
            "serve",
            "--config",
            str(profile_path),
            "--host",
            "127.0.0.1",
            "--port",
            str(args.vllm_port),
            "--enable-prompt-tokens-details",
        ]
        config["vllm_command"] = command
        server = services.launch(
            command,
            output / "vllm.log",
            env={**os.environ, "VLLM_SERVER_DEV_MODE": "1"},
        )
    else:
        server = None
    with profiler.meter.measure("initialization_server_ready_wait", cpu=False):
        models = await ready(
            client, args.vllm_url + "/v1/models", args.startup_timeout, server
        )
    if args.model not in [m["id"] for m in models.get("data", [])]:
        raise RuntimeError("Inference server does not serve the requested model")
    atomic_json(output / "server_models.json", models)
    served = next((m for m in models["data"] if m["id"] == args.model))
    args.max_context_tokens = int(
        served.get("max_model_len") or profile["max-model-len"]
    )
    config["max_context_tokens"] = args.max_context_tokens
    if not args.skip_benchmark_prewarm:
        config["benchmark_prewarm"] = await prewarm_benchmark(args, client, output)
        atomic_json(output / "config.json", config)
