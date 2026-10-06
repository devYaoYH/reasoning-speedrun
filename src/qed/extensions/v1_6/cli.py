"""V1.6 uses four fresh samples; continuations have a separate token budget."""

import argparse
import asyncio
import sys
import signal
import hashlib
import json
from pathlib import Path
from qed.cli import parse_args as canonical_args
from qed.lib.common import PACKAGE
from qed.lib.services import attempt_lock
from qed.lib.policy_runtime import run_policy
from qed.lib.metadata import build_metadata as base_metadata
from .integrity import verify_core
from .policy import run_speedrun

RUNNER_ID = "runner_core_v1_6"


def parse_args(argv=None):
    values = sys.argv[1:] if argv is None else argv
    policy_path = Path(__file__).with_name("policy.json")
    raw = policy_path.read_bytes()
    defaults = json.loads(raw)
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--max-fresh-samples-per-question",
        "--max-attempts-per-question",
        type=int,
        default=defaults["max_fresh_samples_per_question"],
        dest="max_fresh_samples_per_question",
    )
    parser.add_argument(
        "--max-rollout-tokens",
        "--max-tokens",
        type=int,
        default=defaults["max_rollout_tokens"],
        dest="max_rollout_tokens",
    )
    parser.add_argument("--model-profile", default="vllm-v1_6-long64k.yaml")
    parser.add_argument(
        "--max-concurrent-requests",
        type=int,
        default=defaults["max_concurrent_requests"],
    )
    options, remaining = parser.parse_known_args(values)
    forbidden = {
        "--schedule",
        "--max-rounds",
        "--rollouts",
        "--parallelism",
        "--no-continuation",
    }
    if any(value.split("=")[0] in forbidden for value in remaining):
        parser.error(
            "V1.6 uses a slot pool; tune --max-concurrent-requests and --max-fresh-samples-per-question"
        )
    if not 1 <= options.max_fresh_samples_per_question <= 4:
        parser.error("Fresh samples per question must be between one and four")
    if not 1 <= options.max_rollout_tokens <= 65536:
        parser.error("Cumulative rollout output budget must be between one and 65,536")
    if (
        options.max_concurrent_requests is not None
        and options.max_concurrent_requests <= 0
    ):
        parser.error("Concurrent requests must be positive")
    if "--help" in remaining or "-h" in remaining:
        print(__doc__ + "\n" + parser.format_help())
        print(
            "Pool slots default to selected question count. Shared service, dataset, sampling, prompt and benchmark flags are described by `qed --help`. Its round and request-cap controls do not apply here."
        )
        raise SystemExit(0)
    args = canonical_args(["--model-profile", options.model_profile] + remaining)
    vars(args).update(vars(options))
    args.max_tokens = args.max_rollout_tokens
    args.max_attempts_per_question = None
    args.max_rounds = None
    args.schedule = "coverage_barrier_then_pool"
    args.initial_coverage_barrier = True
    args.policy_defaults_file = str(policy_path)
    args.policy_defaults_sha256 = hashlib.sha256(raw).hexdigest()
    args.token_budget_scope = "Initial 8K then one exact-ID continuation for remaining cumulative output, clipped to served total context"
    args.continuation_policy = "one long continuation after the initial capped request"
    return args


async def prepare_problems(problems, args, client):
    args.max_concurrent_requests = args.max_concurrent_requests or len(problems)
    if args.max_concurrent_requests < len(problems):
        raise ValueError("Slot pool must fit one initial request per selected question")
    args.segment_seed_stride = (max(p["problem_idx"] for p in problems) + 1) * 4
    prepared = await asyncio.gather(
        *(prepare_prompt(problem, args, client) for problem in problems)
    )
    args.served_prompt_tokens = {
        str(p["problem_idx"]): p["prompt_tokens"] for p in prepared
    }
    return prepared


async def prepare_prompt(problem, args, client):
    payload = dict(
        model=args.model,
        messages=[
            dict(role="system", content=args.system_prompt),
            dict(role="user", content=problem["problem"]),
        ],
        add_generation_prompt=True,
    )
    if args.disable_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    response = await client.post(args.vllm_url + "/tokenize", json=payload)
    response.raise_for_status()
    count = response.json()["count"]
    if type(count) is not int or not 0 < count < args.max_context_tokens:
        raise ValueError("Invalid served chat-template prompt length")
    return {**problem, "prompt_tokens": count}


def metadata(folder, config):
    value = base_metadata(folder, config)
    hp = value["controls"]["hyperparameters"]
    hp.update(
        max_attempts_per_question=None,
        max_rounds=None,
        max_fresh_samples_per_question=config["max_fresh_samples_per_question"],
        max_rollout_tokens=config["max_rollout_tokens"],
        max_concurrent_requests=config["max_concurrent_requests"],
        continuation_enabled=True,
        schedule="coverage_barrier_then_pool",
        initial_coverage_barrier=True,
        cap_scope="fresh samples only; continuation segments do not consume fresh allowance",
    )
    return value


async def execute(args):
    asyncio.get_running_loop().add_signal_handler(
        signal.SIGTERM, asyncio.current_task().cancel
    )
    await run_policy(
        args,
        policy=run_speedrun,
        prepare_problems=prepare_problems,
        verify=verify_core,
        runner_id=RUNNER_ID,
        runner_module="qed.extensions.v1_6",
        metadata_builder=metadata,
    )


def main(argv=None):
    verify_core(PACKAGE)
    args = parse_args(argv)
    with attempt_lock():
        asyncio.run(execute(args))
