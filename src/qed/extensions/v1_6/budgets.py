"""Fresh samples and continuation segments have independent, finite budgets."""

from dataclasses import dataclass
from qed.lib.continuations import continuation_prefix


@dataclass(frozen=True)
class Plan:
    fresh_sample: int
    segment: int
    generated_before: int
    max_tokens: int
    continuation: object = None


def request_bound(args):
    first = min(args.first_pass_max_tokens, args.max_rollout_tokens)
    return args.max_fresh_samples_per_question * (
        1 if first == args.max_rollout_tokens else 2
    )


def first_plan(problem, args, fresh_sample):
    budget = min(
        args.first_pass_max_tokens,
        args.max_rollout_tokens,
        args.max_context_tokens - problem["prompt_tokens"],
    )
    if budget <= 0:
        raise RuntimeError("Question prompt leaves no generation context")
    return Plan(fresh_sample, 1, 0, budget)


def followup(plan, record, args, folder, artifacts):
    if record["status"] == "error":
        raise RuntimeError(record.get("error", "Generation failed"))
    if record["status"] != "completed" or record["finish_reason"] != "length":
        return None
    if plan.segment != 1:
        return None
    generated = plan.generated_before + record["generated_token_ids_count"]
    remaining = args.max_rollout_tokens - generated
    if remaining <= 0:
        return None
    prefix = continuation_prefix(
        {"rollouts": [record]}, folder, args.max_context_tokens, artifacts
    )
    if prefix is None:
        return None
    budget = min(remaining, prefix["remaining_context"])
    return Plan(plan.fresh_sample, plan.segment + 1, generated, budget, prefix)
