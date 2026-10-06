"""Cheap, ungraded inference warmup outside official timing."""

import asyncio
import time
from reasoning_speedrun.lib.common import utc_now


async def warm_request(args, client, batch_size, tokens, slot):
    request = {
        "model": args.model,
        "messages": [{"role": "user", "content": "Compute 1 + 1."}],
        "max_tokens": tokens,
        "min_tokens": tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed - batch_size + slot,
        "stream": False,
    }
    if args.disable_thinking:
        request["chat_template_kwargs"] = {"enable_thinking": False}
    response = await client.post(args.vllm_url + "/v1/chat/completions", json=request)
    response.raise_for_status()
    return {"request": request, "response": response.json()}


async def warm_inference(args, client, question_count):
    """Warm the configured sampling path at the attempt's maximum batch size."""
    batch_size = min(args.parallelism, question_count) * args.rollouts
    tokens = min(32, args.max_tokens)
    started, start = (utc_now(), time.perf_counter())
    print(f"Inference warmup: {batch_size} streams, {tokens} tokens each", flush=True)
    tasks = [
        asyncio.create_task(warm_request(args, client, batch_size, tokens, slot))
        for slot in range(batch_size)
    ]
    try:
        responses = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done() and (not task.cancelling()):
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    return {
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "latency_s": time.perf_counter() - start,
        "batch_size": batch_size,
        "tokens_per_request": tokens,
        "requests": responses,
    }
