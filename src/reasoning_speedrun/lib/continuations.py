"""Resume capped trajectories only from complete exact token IDs."""

from reasoning_speedrun.lib.storage import AttemptArtifacts


def continuation_prefix(previous, folder, context_limit, artifacts=None):
    """Only continue a capped trajectory with complete, exact token-ID evidence."""
    artifacts = artifacts or AttemptArtifacts(folder.parent.parent)
    if not previous or not previous["rollouts"]:
        return None
    last = previous["rollouts"][-1]
    if last["status"] != "completed" or last["finish_reason"] != "length":
        return None
    token_file = folder / f"rollout-{last['rollout']:02d}" / "tokens.json"
    if not artifacts.has_json(token_file):
        raise RuntimeError(
            "Cannot continue capped output without saved exact token IDs"
        )
    tokens = artifacts.read_json(token_file)
    prompt, output = (tokens["prompt_token_ids"], tokens["output_token_ids"])
    if not prompt or not output or (not tokens["complete"]):
        raise RuntimeError("Incomplete token-ID evidence for continuation")
    prefix = prompt + output
    if len(prefix) >= context_limit - 1:
        return None
    return {
        "prompt": prefix,
        "parent_rollout": last["rollout"],
        "visible_text": tokens["visible_text"],
        "remaining_context": context_limit - len(prefix),
    }
