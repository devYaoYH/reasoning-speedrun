"""V1 sampling payloads for fresh chat and exact-token continuation requests."""


def generation_request(problem, args, rollout, continuation=None):
    request = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": problem["problem"]},
        ],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed
        + problem["problem_idx"] * args.max_attempts_per_question
        + rollout,
        "stream": True,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
        "return_token_ids": True,
    }
    if args.disable_thinking:
        request["chat_template_kwargs"] = {"enable_thinking": False}
    if continuation:
        request.pop("messages")
        request.pop("chat_template_kwargs", None)
        request.update(
            prompt=continuation["prompt"],
            max_tokens=min(args.max_tokens, continuation["remaining_context"]),
        )
        return "/v1/completions", request
    return "/v1/chat/completions", request
