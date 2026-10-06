"""One streaming inference request, token evidence and timing; no scheduling policy."""

import asyncio
import json
import time
from qed.lib.common import utc_now
from qed.lib.metrics import Meter
from qed.lib.transport import sse_payloads
from qed.lib.extraction import CandidateDetector
from qed.lib.requests import generation_request


class RolloutGenerator:

    def __init__(self, question, rollout):
        self.question, self.rollout = (question, rollout)

    def detect(self, part, value, eof=False):
        with self.scope.measure("candidate_parse_enqueue"):
            for event in self.detector.feed(part, value, eof=eof):
                self.question.propose(event, self.rollout)

    def parse_chunk(self, body, elapsed):
        if body.get("error"):
            raise RuntimeError(f"vLLM stream error: {body['error']}")
        if body.get("prompt_token_ids") is not None:
            self.prompt_token_ids = body["prompt_token_ids"]
        if body.get("usage"):
            self.record["usage"] = body["usage"]
        for choice in body.get("choices", []):
            if choice.get("index", 0) != 0:
                continue
            if choice.get("prompt_token_ids") is not None:
                self.prompt_token_ids = choice["prompt_token_ids"]
            if choice.get("token_ids"):
                self.output_token_ids.extend(choice["token_ids"])
            delta = choice.get("delta") or {}
            if self.endpoint == "/v1/completions":
                delta = {"content": choice.get("text", "")}
            for part, value in [
                ("reasoning", delta.get("reasoning_content") or delta.get("reasoning")),
                ("content", delta.get("content")),
            ]:
                if isinstance(value, str) and value:
                    if self.record["ttft_s"] is None:
                        self.record["ttft_s"] = elapsed
                    self.record["last_token_s"] = elapsed
                    self.parts[part].append(value)
                    self.detect(part, value)
            if choice.get("finish_reason"):
                self.record["finish_reason"] = choice["finish_reason"]

    def save_generation(self):
        self.scope.inc("generation_" + self.record["status"])
        ended = time.perf_counter()
        self.record.update(
            generation_finished_at_utc=utc_now(),
            generation_end_monotonic_s=ended,
            generation_latency_s=ended - self.began,
            generation_censored=self.record["status"] != "completed"
            or self.record["finish_reason"] == "length",
        )
        visible_text = "".join(self.parts["reasoning"]) + "".join(self.parts["content"])
        usage = self.record["usage"] or {}
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        self.record.update(
            generated_token_ids_count=len(self.output_token_ids),
            prompt_token_ids_count=(
                len(self.prompt_token_ids) if self.prompt_token_ids else None
            ),
            cached_prompt_tokens=cached,
            prefix_cache_hit_fraction=(
                cached / len(self.prompt_token_ids)
                if cached is not None and self.prompt_token_ids
                else None
            ),
        )
        complete_ids = bool(
            self.prompt_token_ids
            and self.output_token_ids
            and (usage.get("completion_tokens") == len(self.output_token_ids))
        )
        self.question.write_json(
            self.path / "tokens.json",
            {
                "prompt_token_ids": self.prompt_token_ids,
                "output_token_ids": self.output_token_ids,
                "complete": complete_ids,
                "visible_text": (
                    self.continuation["visible_text"] if self.continuation else ""
                )
                + visible_text,
            },
            self.scope,
        )
        self.question.write_json(
            self.path / "response.json",
            {part: "".join(text) for part, text in self.parts.items()},
            self.scope,
        )
        self.record["overhead"] = self.scope.snapshot()
        self.question.write_json(self.path / "telemetry.json", self.record, self.scope)

    async def run(self):
        self.scope = self.question.rollout_meters[self.rollout] = Meter(
            self.question.meter.enabled, parent=self.question.meter
        )
        self.scope.inc("generation_requests")
        self.path = self.question.folder / f"rollout-{self.rollout:02d}"
        if not self.question.artifacts.buffered:
            self.path.mkdir()
        self.continuation = (
            self.question.continuations[self.rollout - self.question.rollout_offset - 1]
            if self.question.continuations is not None
            else self.question.next_continuation
        )
        self.endpoint, self.request = generation_request(
            self.question.problem, self.question.args, self.rollout, self.continuation
        )
        self.question.write_json(self.path / "request.json", self.request, self.scope)
        self.began = time.perf_counter()
        self.record = {
            "rollout": self.rollout,
            "round": self.question.round_no,
            "started_at_utc": utc_now(),
            "start_monotonic_s": self.began,
            "ttft_s": None,
            "last_token_s": None,
            "finish_reason": None,
            "status": "streaming",
            "done_received": False,
            "usage": None,
            "endpoint": self.endpoint,
            "continuation_of_rollout": (
                self.continuation["parent_rollout"] if self.continuation else None
            ),
            "requested_max_tokens": self.request["max_tokens"],
        }
        self.detector = CandidateDetector()
        self.parts = {"reasoning": [], "content": []}
        self.prompt_token_ids = (
            list(self.continuation["prompt"]) if self.continuation else None
        )
        self.output_token_ids = []
        if self.continuation:
            self.detector.feed("content", self.continuation["visible_text"])
        self.question.records.append((self.path, self.record, self.parts))
        try:
            with self.question.artifacts.open_jsonl(
                self.path / "stream.jsonl", "w"
            ) as stream_file:
                async with self.question.client.stream(
                    "POST",
                    self.question.args.vllm_url + self.endpoint,
                    json=self.request,
                    headers={
                        "X-Request-Id": f"{self.question.output.name}-q{self.question.index}-r{self.rollout}"
                    },
                ) as response:
                    self.record["http_status"] = response.status_code
                    self.record["headers_received_s"] = time.perf_counter() - self.began
                    response.raise_for_status()
                    async for payload in sse_payloads(response.aiter_lines()):
                        elapsed = time.perf_counter() - self.began
                        if self.scope.enabled:
                            self.scope.inc("sse_chunks")
                            self.scope.inc("sse_payload_bytes", len(payload.encode()))
                        with self.scope.measure("stream_trace_write_flush"):
                            stream_file.append(
                                {
                                    "elapsed_s": elapsed,
                                    "timestamp_utc": utc_now(),
                                    "data": payload,
                                }
                            )
                        if payload == "[DONE]":
                            self.record["done_received"] = True
                            break
                        with self.scope.measure("sse_json_decode"):
                            body = json.loads(payload)
                        self.parse_chunk(body, elapsed)
                    if self.record["finish_reason"] in ("error", "abort"):
                        raise RuntimeError(
                            f"vLLM terminated with {self.record['finish_reason']}"
                        )
                    if self.record["done_received"] or self.record["finish_reason"] in (
                        "stop",
                        "length",
                    ):
                        # A cap may split an integer. Flush partial clauses only at natural EOF.
                        if self.record["finish_reason"] != "length":
                            for part in self.parts:
                                self.detect(part, "", eof=True)
                        self.record["status"] = "completed"
                    else:
                        raise RuntimeError(
                            "Stream ended without DONE or a terminal finish reason"
                        )
        except asyncio.CancelledError:
            self.record["status"] = "cancelled"
            raise
        except Exception as exc:
            self.record.update(status="error", error=f"{type(exc).__name__}: {exc}")
        finally:
            self.save_generation()
