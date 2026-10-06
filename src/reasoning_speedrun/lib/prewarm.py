"""Ungraded AIME 2024 workload. Never imports or calls the verifier."""

import asyncio
import hashlib
import json
import time
from reasoning_speedrun.lib.aime import benchmark_paths, load_questions
from reasoning_speedrun.lib.common import atomic_json, utc_now


async def reset_cache(client, url, attempts=75):
    """Drop completed prefixes without forcing a reset of running requests."""
    for _ in range(attempts):
        response = await client.post(url + "/reset_prefix_cache", timeout=10)
        response.raise_for_status()
        if response.json().get("success") is True:
            return {"success": True, "at_utc": utc_now()}
        await asyncio.sleep(0.2)
    raise RuntimeError("Prefix cache still held by outstanding requests")


async def prewarm_request(args, client, semaphore, problem):
    request = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": problem["problem"]},
        ],
        "max_tokens": args.prewarm_max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed + 1000000 + problem["problem_idx"],
        "stream": False,
        "return_token_ids": True,
    }
    if args.disable_thinking:
        request["chat_template_kwargs"] = {"enable_thinking": False}
    async with semaphore:
        start, at = (time.perf_counter(), utc_now())
        response = await client.post(
            args.vllm_url + "/v1/chat/completions", json=request
        )
        response.raise_for_status()
        body = response.json()
        if not body.get("choices"):
            raise RuntimeError("Warmup response has no generation")
        row = {
            "problem_idx": problem["problem_idx"],
            "started_at_utc": at,
            "finished_at_utc": utc_now(),
            "latency_s": time.perf_counter() - start,
            "usage": body.get("usage"),
            "finish_reason": body["choices"][0].get("finish_reason"),
        }
        print(
            f"PREWARM Q{problem['problem_idx']:02d} completed in {row['latency_s']:.2f}s",
            flush=True,
        )
        return (row, request, body)


async def prewarm_benchmark(args, client, output):
    prompts, _, source_path = benchmark_paths(2024)
    source = json.loads(source_path.read_text())
    digest = hashlib.sha256(prompts.read_bytes()).hexdigest()
    if digest != source["dataset_sha256"]:
        raise RuntimeError("AIME 2024 warmup prompts differ from the pinned dataset")
    problems = load_questions(year=2024)
    before = await reset_cache(client, args.vllm_url)
    began, started = (time.perf_counter(), utc_now())
    semaphore = asyncio.Semaphore(args.parallelism)
    print(
        f"AIME 2024 ungraded warmup: {len(problems)} questions, one sample each, {args.prewarm_max_tokens} token cap",
        flush=True,
    )
    tasks = [
        asyncio.create_task(prewarm_request(args, client, semaphore, p))
        for p in problems
    ]
    try:
        rows = await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            if not task.done() and (not task.cancelling()):
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    generation_latency = time.perf_counter() - began
    after = await reset_cache(client, args.vllm_url)
    for row, request, response in rows:
        folder = output / "prewarming-2024" / "trace" / f"{row['problem_idx']:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        atomic_json(folder / "request.json", request)
        atomic_json(folder / "response.json", response)
    result = {
        "year": 2024,
        "role": "prewarming",
        "questions": len(problems),
        "source": source["source"],
        "revision": source["revision"],
        "prompt_sha256": digest,
        "started_at_utc": started,
        "finished_at_utc": utc_now(),
        "generation_latency_s": generation_latency,
        "total_latency_s": time.perf_counter() - began,
        "max_tokens": args.prewarm_max_tokens,
        "parallelism": args.parallelism,
        "generation_requests": len(rows),
        "grader_queries": 0,
        "correctness_evaluated": False,
        "prefix_cache_reset_before": before,
        "prefix_cache_reset_after": after,
        "scope": "Ungraded workload before grader launch and official timing; all 30 generations finish naturally or at cap; no AIME 2025 solution prefixes retained",
        "rollouts": [row for row, _, _ in rows],
    }
    atomic_json(output / "prewarm.json", result)
    return result
