"""Describe canonical attempt interventions and reproducible controls.

Run `python -m qed.lib.metadata --all` after importing new attempts, or pass
one attempt directory. It creates missing metadata.json files from recorded
config/profile evidence; edit the intervention and comparison notes afterward.
Existing annotations are validated and preserved. No inference is launched.
"""

from __future__ import annotations

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

from jsonschema import Draft202012Validator
import yaml

from qed.lib.common import PACKAGE, atomic_json, workdir
from qed.lib.datasets import recorded_dataset

SCHEMA = PACKAGE / "metadata.schema.json"


@lru_cache(maxsize=1)
def validator():
    schema = json.loads(SCHEMA.read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_metadata(value, attempt_id):
    errors = sorted(validator().iter_errors(value), key=lambda e: str(e.path))
    if errors:
        raise ValueError(f"Metadata schema: {errors[0].message}")
    if value["attempt_id"] != attempt_id:
        raise ValueError("Metadata attempt_id does not match directory")
    if value["intervention"]["reference_attempt_id"] == attempt_id:
        raise ValueError("An attempt cannot be its own comparison reference")
    dataset = value["provenance"].get("dataset")
    if dataset and value["controls"]["dataset"] != dataset_label(dataset):
        raise ValueError("Metadata dataset identity/year disagree")
    return value


def dataset_label(dataset):
    return f"AIME {dataset['year']}" if dataset["year"] else dataset["id"]


def build_metadata(folder, config=None):
    """Normalize measured settings; unknown values remain null, never guessed."""
    config = config or json.loads((folder / "config.json").read_text())
    profile_file = folder / "model_profile.json"
    profile = json.loads(profile_file.read_text()) if profile_file.exists() else {}
    launch = (
        config.get("launch_profile") or yaml.safe_load(profile.get("yaml", "")) or {}
    )
    dataset = recorded_dataset(config)
    model = config.get("model")
    quant = launch.get("quantization")
    quant_source = "launch_profile" if quant else "unrecorded"
    if not quant and model and "GPTQ" in model:
        quant, quant_source = "GPTQ Int4", "model identifier; backend inferred by vLLM"
    elif (
        not quant
        and launch.get("dtype") in ("bfloat16", "float16")
        and model
        and not re.search(r"(?i)int[48]|fp[48]|gptq|awq", model)
    ):
        quant, quant_source = "none", "recorded unquantized model and launch dtype"
    revision = re.search(
        r"(?i)(?:revision[: ]+)\s*([0-9a-f]{40})", profile.get("yaml", "")
    )
    strategy = config.get("strategy", "parallel_streaming")
    coverage = strategy == "coverage" or str(config.get("runner_id", "")).startswith(
        "speedrun_v"
    )
    sampling_keys = (
        "parallelism",
        "rollouts",
        "temperature",
        "top_p",
        "seed",
        "disable_thinking",
        "question_timeout",
        "schedule",
        "warmup_batch_size",
        "no_overhead_profile",
        "overhead_interval",
        "engine_metrics_interval",
        "benchmark",
        "buffer_traces",
        "no_gpu_telemetry",
        "token_budgets",
        "max_concurrent_requests",
        "seed_stride",
        "budget_mode",
        "initial_rollouts",
        "expansion_trigger",
        "token_budget_scope",
        "skip_benchmark_prewarm",
        "prewarm_max_tokens",
    )
    hp = {key: config[key] for key in sampling_keys if key in config}
    hp.update(
        strategy=strategy,
        max_tokens=config.get("max_tokens"),
        first_pass_max_tokens=(
            config.get("first_pass_max_tokens", 8192)
            if coverage
            else config.get("max_tokens")
        ),
        max_attempts_per_question=(
            config.get("max_attempts_per_question", 4)
            if coverage
            else config.get("rollouts")
        ),
        max_rounds=config.get("max_rounds", 4) if coverage else 1,
        continuation_enabled=coverage and not config.get("no_continuation", False),
        answer_extraction=config.get("answer_extraction")
        or (
            "completed final box only"
            if strategy == "baseline"
            else "streaming prospective candidates"
        ),
        system_prompt_sha256=(
            hashlib.sha256(config["system_prompt"].encode()).hexdigest()
            if config.get("system_prompt")
            else None
        ),
    )
    gpu_samples = folder / "gpu.jsonl"
    total = None
    if gpu_samples.exists():
        with gpu_samples.open() as stream:
            for line in stream:
                if line.strip():
                    total = json.loads(line).get("vram_total_mib")
                    if total is not None:
                        break
    value = {
        "$schema": "https://github.com/devYaoYH/reasoning-speedrun/blob/main/src/qed/metadata.schema.json",
        "schema_version": 1,
        "attempt_id": folder.name,
        "label": model or folder.name,
        "intervention": {
            "label": "Not annotated",
            "reference_attempt_id": None,
            "changed_variables": [],
            "comparison_note": "No intervention comparison has been annotated.",
        },
        "runner": {
            "module": config.get("runner_module") or "qed",
            "version": config.get("runner_id")
            or f"git:{config.get('git_commit', 'unknown')}",
            "git_commit": config.get("git_commit"),
            "git_dirty": config.get("git_dirty"),
        },
        "model": {
            "id": model,
            "revision": revision.group(1) if revision else None,
            "quantization": quant,
            "quantization_source": quant_source,
            "activation_dtype": launch.get("dtype"),
            "kv_cache_dtype": launch.get("kv-cache-dtype"),
            "linear_backend": launch.get("linear-backend"),
        },
        "gpu": {
            "device": "simulated" if config.get("simulated") else None,
            "device_count": launch.get("tensor-parallel-size"),
            "total_vram_mib": total,
            "memory_utilization": launch.get("gpu-memory-utilization"),
            "configured_envelope_mib": (
                total * launch["gpu-memory-utilization"]
                if total is not None
                and launch.get("gpu-memory-utilization") is not None
                else None
            ),
        },
        "controls": {
            "dataset": dataset_label(dataset),
            "question_indices": config.get("question_indices")
            or config.get("questions")
            or [],
            "target_correct": config.get("target_correct", 18),
            "grader_cost_s": config.get("grader_cost"),
            "grader_serial": True,
            "max_context_tokens": launch.get(
                "max-model-len", config.get("max_context_tokens")
            ),
            "max_num_seqs": launch.get("max-num-seqs"),
            "prefix_caching": launch.get("enable-prefix-caching"),
            "reasoning_parser": launch.get("reasoning-parser"),
            "hyperparameters": hp,
        },
        "provenance": {
            "dataset": dataset,
            "config": "config.json",
            "profile": "model_profile.json" if profile else None,
            "profile_sha256": config.get("model_profile_sha256")
            or profile.get("sha256"),
            "notes": [
                "Normalized from saved config/profile and GPU samples. Null means unrecorded."
            ],
        },
    }
    if config.get("simulated"):
        # A simulated attempt must never be mistaken for a measurement on hardware.
        value["simulation"] = config.get("simulation")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path, nargs="?")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Create/validate metadata for every local canonical attempt",
    )
    args = parser.parse_args()
    if bool(args.attempt) == args.all:
        parser.error("Choose an attempt directory or --all")
    folders = (
        sorted((workdir() / "attempts").glob("*/config.json"))
        if args.all
        else [args.attempt / "config.json"]
    )
    for config_path in folders:
        folder = config_path.parent
        path = folder / "metadata.json"
        value = (
            json.loads(path.read_text()) if path.exists() else build_metadata(folder)
        )
        changed = False
        if path.exists() and "dataset" not in value.get("provenance", {}):
            validate_metadata(value, folder.name)
            derived = build_metadata(folder)
            if value["controls"]["dataset"] != derived["controls"]["dataset"]:
                raise ValueError(
                    "Saved metadata dataset disagrees with recorded configuration"
                )
            value["provenance"]["dataset"] = derived["provenance"]["dataset"]
            changed = True
        validate_metadata(value, folder.name)
        if changed:
            atomic_json(path, value)
            print(f"Added dataset provenance to {path}; annotations preserved.")
        elif not path.exists():
            atomic_json(path, value)
            print(
                f"Created {path}; annotate intervention and reference before comparing."
            )
        else:
            print(f"Validated {path}")


if __name__ == "__main__":
    main()
