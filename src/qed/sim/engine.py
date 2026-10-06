"""The simulated serving engine: admission, prefill/decode timing, prefix cache, metrics.

Deliberately coarse. ``StaticRates`` gives every running request the same decode
and prefill speed regardless of load; requests beyond ``max_num_seqs`` wait.
Finer models (batch-dependent decode, memory-bandwidth and arithmetic-intensity
estimates, KV-capacity admission and preemption) replace ``StaticRates`` or extend
``Engine.admit`` without touching the transport or the runner.
"""

import asyncio
from array import array
from collections import namedtuple
import hashlib
import time

from qed.sim.model import build_plan, plan_rng

Load = namedtuple("Load", "running waiting live_tokens")
Step = namedtuple("Step", "tokens emitted finish prompt_ids cached first")


class StaticRates:
    """Load-independent per-request rates. The interface a finer model implements."""

    def __init__(self, config):
        self.config = config

    def prefill_seconds(self, uncached_tokens, load):
        return uncached_tokens / self.config.prefill_tps

    def decode_seconds(self, tokens, load):
        return tokens / self.config.decode_tps


def digest(ids, seed=b""):
    return hashlib.blake2b(array("I", ids).tobytes() + seed, digest_size=16).digest()


class PrefixCache:
    """vLLM-style chained block hashes: a request reuses its longest cached block prefix."""

    def __init__(self, block):
        self.block, self.blocks = block, set()

    def chain(self, ids):
        previous = b""
        for start in range(0, len(ids) - self.block + 1, self.block):
            previous = digest(ids[start : start + self.block], previous)
            yield previous

    def match(self, ids):
        hits = 0
        for h in self.chain(ids):
            if h not in self.blocks:
                break
            hits += 1
        # vLLM never serves a whole prompt from cache; one token is always computed.
        return min(hits * self.block, max(len(ids) - 1, 0))

    def insert(self, ids):
        self.blocks.update(self.chain(ids))

    def clear(self):
        self.blocks.clear()


class Engine:
    def __init__(self, config, model, key, *, max_model_len, max_new_tokens=None,
                 gpu_memory_utilization=0.9, rates=None):
        self.config, self.model, self.key = config, model, key
        self.max_model_len, self.max_new_tokens = max_model_len, max_new_tokens
        self.gpu_memory_utilization = gpu_memory_utilization
        self.rates = rates or StaticRates(config)
        self.slots = asyncio.Semaphore(config.max_num_seqs)
        self.cache = PrefixCache(config.block_tokens)
        self.sequences = {}  # digest(prompt + emitted ids) -> (plan, emitted offset)
        self.running = self.waiting = self.live_tokens = 0
        self.counters = {
            "requests": 0, "prompt_tokens": 0, "generation_tokens": 0,
            "prefix_queries": 0, "prefix_hits": 0, "queue_s": 0.0, "ttft_s": 0.0,
            "prefill_s": 0.0, "decode_s": 0.0, "first_tokens": 0, "cancelled": 0,
            "peak_running": 0, "peak_waiting": 0, "peak_live_tokens": 0,
        }
        self.counter = 0

    @property
    def kv_capacity_tokens(self):
        c = self.config
        usable = c.vram_total_mib * self.gpu_memory_utilization - c.weights_mib
        return max(0, int(usable * 2**20 / c.kv_bytes_per_token))

    def load(self):
        return Load(self.running, self.waiting, self.live_tokens)

    def plan_for(self, text, request_seed, min_tokens=0):
        gold = self.key.get(text.strip())
        self.counter += 1
        seed = request_seed if request_seed is not None else f"auto{self.counter}"
        rng = plan_rng(self.config.behavior.seed, seed, text)
        return build_plan(self.config.behavior, rng, gold, min_tokens)

    def continuation(self, prompt_ids):
        return self.sequences.get(digest(prompt_ids))

    def reset_prefix_cache(self):
        if self.running or self.waiting:
            return False
        self.cache.clear()
        return True

    async def generate(self, prompt_ids, plan, offset, max_tokens):
        """Yield ``Step`` chunks of one request; always release the slot and record the prefix."""
        c, counters = self.config, self.counters
        arrived = time.perf_counter()
        loop = asyncio.get_running_loop()
        blocked = self.slots.locked()  # a free slot is taken without queueing
        if blocked:
            self.waiting += 1
            counters["peak_waiting"] = max(counters["peak_waiting"], self.waiting)
        try:
            await self.slots.acquire()
        finally:
            if blocked:
                self.waiting -= 1
        self.running += 1
        counters["peak_running"] = max(counters["peak_running"], self.running)
        counters["requests"] += 1
        emitted_ids, held = [], len(prompt_ids)
        self.live_tokens += held
        finished = False
        try:
            queued = time.perf_counter() - arrived
            cached = self.cache.match(prompt_ids) if c.prefix_cache else 0
            counters["prefix_queries"] += len(prompt_ids)
            counters["prefix_hits"] += cached
            counters["prompt_tokens"] += len(prompt_ids)
            prefill = self.rates.prefill_seconds(len(prompt_ids) - cached, self.load())
            await asyncio.sleep(prefill)
            limit = min(max_tokens, plan.total - offset)
            decode_started = deadline = loop.time()
            emitted, first = 0, True
            while emitted < limit:
                n = min(c.stream_interval_tokens, limit - emitted)
                deadline += self.rates.decode_seconds(n, self.load())
                await asyncio.sleep(max(0.0, deadline - loop.time()))
                tokens = plan.render(offset + emitted, offset + emitted + n)
                emitted += len(tokens)
                emitted_ids += [t[2] for t in tokens]
                self.live_tokens += len(tokens)
                held += len(tokens)
                counters["generation_tokens"] += len(tokens)
                counters["peak_live_tokens"] = max(counters["peak_live_tokens"], self.live_tokens)
                if first:
                    counters["ttft_s"] += time.perf_counter() - arrived
                    counters["first_tokens"] += 1
                finish = "stop" if offset + emitted >= plan.total else "length" if emitted >= limit else None
                yield Step(tokens, emitted, finish, prompt_ids, cached, first)
                first = False
            counters["queue_s"] += queued
            counters["prefill_s"] += prefill
            counters["decode_s"] += loop.time() - decode_started
            finished = True
        finally:
            if not finished:
                counters["cancelled"] += 1
            sequence = prompt_ids + emitted_ids
            self.sequences[digest(sequence)] = (plan, offset + len(emitted_ids))
            if c.prefix_cache:
                self.cache.insert(sequence)
            self.live_tokens -= held
            self.running -= 1
            self.slots.release()

    def metrics_text(self):
        """Public vLLM gauges/counters, in Prometheus text format."""
        c, labels = self.counters, f'{{engine="0",model_name="{self.model}"}}'
        usage = min(1.0, self.live_tokens / self.kv_capacity_tokens) if self.kv_capacity_tokens else 1.0
        rows = {
            "num_requests_running": self.running,
            "num_requests_waiting": self.waiting,
            "kv_cache_usage_perc": usage,
            "num_preemptions_total": 0,
            "prefix_cache_queries_total": c["prefix_queries"],
            "prefix_cache_hits_total": c["prefix_hits"],
            "prompt_tokens_total": c["prompt_tokens"],
            "generation_tokens_total": c["generation_tokens"],
            "request_queue_time_seconds_sum": c["queue_s"],
            "request_queue_time_seconds_count": c["requests"],
            "request_prefill_time_seconds_sum": c["prefill_s"],
            "request_decode_time_seconds_sum": c["decode_s"],
            "time_to_first_token_seconds_sum": c["ttft_s"],
            "time_to_first_token_seconds_count": c["first_tokens"],
        }
        return "".join(f"vllm:{name}{labels} {float(value)}\n" for name, value in rows.items())

    def report(self):
        c = self.counters
        return {
            **c,
            "prefix_hit_fraction": c["prefix_hits"] / c["prefix_queries"] if c["prefix_queries"] else None,
            "kv_capacity_tokens": self.kv_capacity_tokens,
            "kv_overcommit_peak_tokens": max(0, c["peak_live_tokens"] - self.kv_capacity_tokens),
            "scope": "static-rate simulation; decode speed does not depend on load or KV pressure",
        }
