"""Shared fake-SSE helpers for offline policy tests."""

import asyncio
import json

import httpx


def chunk(text, part="content", finish=None):
    return (
        "data: "
        + json.dumps(
            {"choices": [{"index": 0, "delta": {part: text}, "finish_reason": finish}]}
        )
        + "\n\n"
    ).encode()


class Stream(httpx.AsyncByteStream):
    def __init__(self, chunks, delay=0.002, hang=False):
        self.chunks, self.delay, self.hang = chunks, delay, hang
        self.closed = False

    async def __aiter__(self):
        for value in self.chunks:
            await asyncio.sleep(self.delay)
            yield value
        if self.hang:
            await asyncio.Event().wait()

    async def aclose(self):
        self.closed = True


def capped(prompt, tokens, text, *, completion=False):
    choice = {"index": 0, "token_ids": tokens, "finish_reason": "length"}
    choice.update(
        {"text": text, "prompt_token_ids": prompt}
        if completion
        else {"delta": {"content": text}}
    )
    row = {
        "choices": [choice],
        "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(tokens)},
    }
    if not completion:
        row["prompt_token_ids"] = prompt
    return [("data: " + json.dumps(row) + "\n\n").encode(), b"data: [DONE]\n\n"]


def fake_provenance(year=2025):
    """Dataset evidence without the licensed benchmark files."""
    return {
        "id": f"aime_{year}",
        "year": year,
        "role": "development",
        "source": f"https://huggingface.co/datasets/MathArena/aime_{year}",
        "revision": "0" * 40,
        "split": "train",
        "rows": 30,
        "prompt_path": f"aime_{year}_problems.jsonl",
        "prompt_sha256": "0" * 64,
        "grader_path": f"grader/aime_{year}.jsonl",
        "grader_sha256": "1" * 64,
        "inferred_from_legacy_runner": False,
    }
