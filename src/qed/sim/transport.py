"""In-process stand-in for the vLLM HTTP server, as an httpx transport.

Requests to the simulated server's host:port are answered here (OpenAI-style
chat and completions streams with token IDs and usage, ``/tokenize``,
``/v1/models``, ``/metrics``, ``/reset_prefix_cache``); everything else, notably
the real grader, falls through to a real network transport.
"""

import json
import time

import httpx

from qed.sim.model import chat_prompt_ids, encode


def sse(body):
    return b"data: " + json.dumps(body, ensure_ascii=False).encode() + b"\n\n"


class SimStream(httpx.AsyncByteStream):
    def __init__(self, source):
        self.source = source

    async def __aiter__(self):
        async for chunk in self.source:
            yield chunk

    async def aclose(self):
        await self.source.aclose()


class SimulatedTransport(httpx.AsyncBaseTransport):
    def __init__(self, engine, port, fallback, host="127.0.0.1"):
        self.engine, self.port, self.host, self.fallback = engine, port, host, fallback
        self.requests = 0

    async def aclose(self):
        await self.fallback.aclose()

    async def handle_async_request(self, request):
        if request.url.host != self.host or request.url.port != self.port:
            return await self.fallback.handle_async_request(request)
        path, engine = request.url.path, self.engine
        if path == "/v1/models":
            return self.json(
                {"object": "list", "data": [{"id": engine.model, "object": "model",
                                             "root": engine.model, "max_model_len": engine.max_model_len}]}
            )
        if path == "/metrics":
            return httpx.Response(200, text=engine.metrics_text(), headers={"content-type": "text/plain"})
        if path == "/health":
            return httpx.Response(200)
        if path == "/reset_prefix_cache":
            return self.json({"success": engine.reset_prefix_cache()})
        try:
            body = json.loads(request.content or b"{}")
        except json.JSONDecodeError:
            return self.error("Invalid JSON body")
        if path == "/tokenize":
            ids = chat_prompt_ids(body["messages"]) if "messages" in body else encode(body.get("prompt", ""))
            return self.json({"count": len(ids), "max_model_len": engine.max_model_len, "tokens": ids})
        if path in ("/v1/chat/completions", "/v1/completions"):
            return await self.completion(path, body)
        return self.error(f"Not found: {path}", 404)

    def json(self, value, status=200):
        return httpx.Response(status, json=value)

    def error(self, message, status=400):
        return self.json({"error": {"message": message, "type": "BadRequestError", "code": status}}, status)

    async def completion(self, path, body):
        engine, chat = self.engine, path.endswith("chat/completions")
        if body.get("model") != engine.model:
            return self.error(f"The model `{body.get('model')}` does not exist.", 404)
        max_tokens = body.get("max_tokens") or body.get("max_completion_tokens") or engine.max_model_len
        if engine.max_new_tokens and max_tokens > engine.max_new_tokens:
            max_tokens = engine.max_new_tokens
        if chat:
            messages = body.get("messages") or []
            prompt_ids = chat_prompt_ids(messages)
            user = next((m for m in reversed(messages) if m.get("role") == "user"), {})
            question = user.get("content", "")
            plan = engine.plan_for(question if isinstance(question, str) else json.dumps(question),
                                   body.get("seed"), body.get("min_tokens") or 0)
            offset = 0
        else:
            prompt = body.get("prompt")
            if not isinstance(prompt, list) or not all(isinstance(i, int) for i in prompt):
                return self.error("The simulated backend takes token-ID prompts")
            known = engine.continuation(prompt)
            if known is None:
                return self.error("Unknown token prefix: the simulated backend can only continue a stream it served")
            (plan, offset), prompt_ids = known, prompt
        room = engine.max_model_len - len(prompt_ids)
        if room < 1:
            return self.error(f"Prompt of {len(prompt_ids)} tokens exceeds the context length {engine.max_model_len}")
        max_tokens = min(max_tokens, room)
        rid, created = f"sim-{engine.counter}-{int(time.time() * 1000)}", int(time.time())
        if not body.get("stream"):
            return await self.collect(chat, rid, created, prompt_ids, plan, offset, max_tokens)
        source = self.events(chat, rid, created, prompt_ids, plan, offset, max_tokens, bool(body.get("return_token_ids")))
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=SimStream(source))

    async def collect(self, chat, rid, created, prompt_ids, plan, offset, max_tokens):
        engine, texts, ids, finish, cached = self.engine, {"reasoning": [], "content": []}, [], None, 0
        async for step in engine.generate(prompt_ids, plan, offset, max_tokens):
            for part, text, tid in step.tokens:
                texts[part].append(text)
                ids.append(tid)
            finish, cached = step.finish, step.cached
        usage = self.usage(len(prompt_ids), len(ids), cached)
        if chat:
            message = {"role": "assistant", "content": "".join(texts["content"]) or None}
            if texts["reasoning"]:
                message["reasoning_content"] = "".join(texts["reasoning"])
            choice = {"index": 0, "message": message, "finish_reason": finish}
            return self.json({"id": rid, "object": "chat.completion", "created": created,
                              "model": engine.model, "choices": [choice], "usage": usage})
        text = "".join(texts["reasoning"] + texts["content"])
        return self.json({"id": rid, "object": "text_completion", "created": created, "model": engine.model,
                          "choices": [{"index": 0, "text": text, "finish_reason": finish}], "usage": usage})

    @staticmethod
    def usage(prompt, completion, cached):
        return {"prompt_tokens": prompt, "total_tokens": prompt + completion,
                "completion_tokens": completion, "prompt_tokens_details": {"cached_tokens": cached}}

    async def events(self, chat, rid, created, prompt_ids, plan, offset, max_tokens, token_ids):
        engine = self.engine
        head = {"id": rid, "created": created, "model": engine.model}
        head["object"] = "chat.completion.chunk" if chat else "text_completion"
        total, cached, finish = 0, 0, None
        async for step in engine.generate(prompt_ids, plan, offset, max_tokens):
            total, cached, finish = step.emitted, step.cached, step.finish
            body = {**head, "usage": self.usage(len(prompt_ids), total, cached)}
            if step.first and token_ids and chat:
                body["prompt_token_ids"] = prompt_ids
            ids = [t[2] for t in step.tokens]
            if chat:
                delta = {}
                for part, key in (("reasoning", "reasoning_content"), ("content", "content")):
                    text = "".join(t[1] for t in step.tokens if t[0] == part)
                    if text:
                        delta[key] = text
                choice = {"index": 0, "delta": delta, "finish_reason": None}
            else:
                choice = {"index": 0, "text": "".join(t[1] for t in step.tokens), "finish_reason": None}
                if step.first and token_ids:
                    choice["prompt_token_ids"] = prompt_ids
            if token_ids:
                choice["token_ids"] = ids
            body["choices"] = [choice]
            yield sse(body)
        final = {"index": 0, "finish_reason": finish}
        if chat:
            final["delta"] = {}
        else:
            final["text"] = ""
        if token_ids:
            final["token_ids"] = []
        yield sse({**head, "choices": [final], "usage": self.usage(len(prompt_ids), total, cached)})
        yield sse({**head, "choices": [], "usage": self.usage(len(prompt_ids), total, cached)})
        yield b"data: [DONE]\n\n"
