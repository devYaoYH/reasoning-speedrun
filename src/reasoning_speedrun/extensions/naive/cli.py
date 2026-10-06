"""Naive baseline: all samples of all questions at once, final answers only."""

import argparse
import asyncio
import signal
import sys
from reasoning_speedrun.cli import parse_args as canonical_args
from reasoning_speedrun.lib.common import PACKAGE
from reasoning_speedrun.lib.metadata import build_metadata as base_metadata
from reasoning_speedrun.lib.policy_runtime import run_policy
from reasoning_speedrun.lib.services import attempt_lock
from .integrity import verify_core
from .policy import run_speedrun

RUNNER_ID = "runner_core_naive"
FORBIDDEN = {
    "--schedule",
    "--max-rounds",
    "--rollouts",
    "--parallelism",
    "--no-continuation",
    "--max-attempts-per-question",
    "--first-pass-max-tokens",
}


def parse_args(argv=None):
    values = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument(
        "--samples-per-question",
        type=int,
        default=4,
        help="Samples started at once for every question (pass@k fan-out)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=16384, help="Output cap of each sample"
    )
    options, remaining = parser.parse_known_args(values)
    if any(value.split("=")[0] in FORBIDDEN for value in remaining):
        parser.error(
            "The naive policy has no rounds, barriers or continuations; tune "
            "--samples-per-question and --max-tokens"
        )
    if options.samples_per_question < 1 or options.max_tokens < 1:
        parser.error("Samples and token budget must be positive")
    if "--help" in remaining or "-h" in remaining:
        print(__doc__ + "\n" + parser.format_help())
        print(
            "Service, dataset, sampling, prompt and benchmark flags are shared with "
            "`reasoning-speedrun --help`."
        )
        raise SystemExit(0)
    args = canonical_args(remaining)
    k = options.samples_per_question
    args.samples_per_question = k
    args.rollouts = k
    args.max_attempts_per_question = k
    args.max_tokens = args.first_pass_max_tokens = options.max_tokens
    args.max_rounds = 1
    args.no_continuation = True
    args.schedule = "full_fanout"
    # prepare_problems raises parallelism to the question count: every question
    # starts together and concurrency is whatever the server admits.
    return args


async def prepare_problems(problems, args, client):
    args.parallelism = len(problems)
    return problems


def metadata(folder, config):
    value = base_metadata(folder, config)
    hp = value["controls"]["hyperparameters"]
    hp.update(
        max_attempts_per_question=config["samples_per_question"],
        max_rounds=1,
        continuation_enabled=False,
        schedule="full_fanout",
        answer_extraction="completed final box only",
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
        runner_module="reasoning_speedrun.extensions.naive",
        metadata_builder=metadata,
    )


def main(argv=None):
    verify_core(PACKAGE)
    args = parse_args(argv)
    with attempt_lock():
        asyncio.run(execute(args))
