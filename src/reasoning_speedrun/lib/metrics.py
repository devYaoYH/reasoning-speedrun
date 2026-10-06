"""Observational runner profiling; no candidate or scheduling decisions live here."""

from __future__ import annotations

import asyncio
from collections import defaultdict, deque
from contextlib import contextmanager, nullcontext
import json
import resource
import sys
import time

from reasoning_speedrun.lib.common import utc_now
from reasoning_speedrun.lib.storage import AttemptArtifacts


class Meter:
    """Synchronous work uses thread CPU; overlapping async waits use wall time only."""

    def __init__(self, enabled=True, parent=None):
        self.enabled, self.parent = enabled, parent
        self.timings, self.counters, self.maxima = {}, {}, {}

    def observe(self, name, wall_s, cpu_s=None):
        if not self.enabled:
            return
        row = self.timings.setdefault(
            name,
            {
                "count": 0,
                "wall_total_s": 0.0,
                "wall_max_s": 0.0,
                "thread_cpu_total_s": 0.0,
                "cpu_samples": 0,
            },
        )
        row["count"] += 1
        row["wall_total_s"] += wall_s
        row["wall_max_s"] = max(row["wall_max_s"], wall_s)
        if cpu_s is not None:
            row["thread_cpu_total_s"] += cpu_s
            row["cpu_samples"] += 1
        if self.parent:
            self.parent.observe(name, wall_s, cpu_s)

    def measure(self, name, *, cpu=True):
        if not self.enabled:
            return nullcontext()
        return self._measure(name, cpu=cpu)

    @contextmanager
    def _measure(self, name, *, cpu=True):
        start = time.perf_counter()
        cpu_start = time.thread_time() if cpu else None
        try:
            yield
        finally:
            self.observe(
                name,
                time.perf_counter() - start,
                time.thread_time() - cpu_start if cpu else None,
            )

    def inc(self, name, count=1):
        if self.enabled:
            self.counters[name] = self.counters.get(name, 0) + count
            if self.parent:
                self.parent.inc(name, count)

    def high_water(self, name, value):
        if self.enabled:
            self.maxima[name] = max(self.maxima.get(name, 0), value)
            if self.parent:
                self.parent.high_water(name, value)

    def snapshot(self):
        return {
            "enabled": self.enabled,
            "timings": {k: dict(v) for k, v in self.timings.items()},
            "counters": dict(self.counters),
            "maxima": dict(self.maxima),
        }


def merge_meters(previous, current):
    """Aggregate persisted question rounds without recounting them in a parent."""
    if not previous:
        return current
    merged = json.loads(json.dumps(current))
    for name, values in previous.get("timings", {}).items():
        row = merged["timings"].setdefault(name, {k: 0 for k in values})
        for key, value in values.items():
            row[key] = max(row[key], value) if key == "wall_max_s" else row[key] + value
    for name, value in previous.get("counters", {}).items():
        merged["counters"][name] = merged["counters"].get(name, 0) + value
    for name, value in previous.get("maxima", {}).items():
        merged["maxima"][name] = max(merged["maxima"].get(name, 0), value)
    return merged


def parse_engine_metrics(text):
    """Select public vLLM gauges/counters; preserve labels for multi-engine servers."""
    wanted = {
        "num_requests_running",
        "num_requests_waiting",
        "kv_cache_usage_perc",
        "gpu_cache_usage_perc",
        "num_preemptions_total",
        "prefix_cache_hits_total",
        "prefix_cache_queries_total",
        "request_queue_time_seconds_sum",
        "request_queue_time_seconds_count",
        "request_prefill_time_seconds_sum",
        "request_decode_time_seconds_sum",
        "time_to_first_token_seconds_sum",
        "time_to_first_token_seconds_count",
        "generation_tokens_total",
        "prompt_tokens_total",
    }
    rows = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        pieces = line.rsplit(None, 1)
        if len(pieces) != 2:
            continue
        metric = pieces[0]
        name = metric.split("{", 1)[0]
        if name.removeprefix("vllm:") not in wanted:
            continue
        try:
            value = float(pieces[1])
        except ValueError:
            continue
        rows.append({"metric": metric, "name": name, "value": value})
    return rows


class AttemptProfiler:
    def __init__(
        self,
        output,
        *,
        enabled=True,
        interval=0.05,
        engine_interval=1.0,
        artifacts=None,
    ):
        self.output, self.enabled = output, enabled
        self.artifacts = artifacts or AttemptArtifacts(output)
        self.interval, self.engine_interval = interval, engine_interval
        self.meter = Meter(enabled)
        self.phase = "initializing"
        self.started = time.perf_counter()
        self.initial_usage = resource.getrusage(resource.RUSAGE_SELF)
        self.official_usage = None
        self.official_end_usage = None
        self.first_submit_s = None
        self.lag = defaultdict(
            lambda: {
                "count": 0,
                "sum_s": 0.0,
                "max_s": 0.0,
                "recent": deque(maxlen=10000),
            }
        )
        self.tasks = []
        self.engine_samples = 0
        self.engine_errors = []
        self.engine_last = {}
        self.engine_max = {}
        self.engine_first = {}

    def official_start(self, client, base_url):
        self.phase = "official"
        self.official_usage = resource.getrusage(resource.RUSAGE_SELF)
        if self.enabled:
            self.tasks = [
                asyncio.create_task(self._lag_loop()),
                asyncio.create_task(self._engine_loop(client, base_url)),
            ]

    def submitted(self, elapsed):
        if self.enabled and (
            self.first_submit_s is None or elapsed < self.first_submit_s
        ):
            self.first_submit_s = elapsed

    async def _lag_loop(self):
        while True:
            expected = time.perf_counter() + self.interval
            await asyncio.sleep(self.interval)
            lag = max(0.0, time.perf_counter() - expected)
            row = self.lag[self.phase]
            row["count"] += 1
            row["sum_s"] += lag
            row["max_s"] = max(row["max_s"], lag)
            row["recent"].append(lag)

    async def _engine_loop(self, client, base_url):
        with self.artifacts.open_jsonl(
            self.output / "inference_metrics.jsonl", "w"
        ) as file:
            while True:
                try:
                    start = time.perf_counter()
                    response = await client.get(base_url + "/metrics", timeout=2)
                    response.raise_for_status()
                    self.meter.observe(
                        "engine_metrics_http_wait", time.perf_counter() - start
                    )
                    with self.meter.measure("engine_metrics_parse_write"):
                        rows = parse_engine_metrics(response.text)
                        event = {
                            "timestamp_utc": utc_now(),
                            "monotonic_s": time.perf_counter(),
                            "phase": self.phase,
                            "metrics": rows,
                        }
                        file.append(event)
                    self.engine_samples += 1
                    for row in rows:
                        key, value = row["metric"], row["value"]
                        self.engine_first.setdefault(key, value)
                        self.engine_last[key] = value
                        self.engine_max[key] = max(
                            self.engine_max.get(key, value), value
                        )
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    self.engine_errors.append(f"{type(exc).__name__}: {exc}")
                    self.engine_errors = self.engine_errors[-10:]
                    self.meter.inc("engine_metrics_poll_errors")
                await asyncio.sleep(self.engine_interval)

    async def stop(self):
        for task in self.tasks:
            task.cancel()
        results = await asyncio.gather(*self.tasks, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                self.engine_errors.append(f"{type(result).__name__}: {result}")
        self.tasks.clear()
        if self.official_usage and self.official_end_usage is None:
            self.official_end_usage = resource.getrusage(resource.RUSAGE_SELF)

    def snapshot(self):
        usage = resource.getrusage(resource.RUSAGE_SELF)
        lag = {}
        for phase, row in self.lag.items():
            recent = sorted(row["recent"])
            lag[phase] = {
                "count": row["count"],
                "mean_s": row["sum_s"] / row["count"],
                "max_s": row["max_s"],
                "recent_p95_s": recent[int((len(recent) - 1) * 0.95)],
                "p95_sample_count": len(recent),
            }
        cpu = {
            "user_s": usage.ru_utime - self.initial_usage.ru_utime,
            "system_s": usage.ru_stime - self.initial_usage.ru_stime,
            "peak_rss_mib": usage.ru_maxrss
            / (2**20 if sys.platform == "darwin" else 1024),
        }
        if self.official_usage:
            end = self.official_end_usage or usage
            cpu.update(
                official_user_s=end.ru_utime - self.official_usage.ru_utime,
                official_system_s=end.ru_stime - self.official_usage.ru_stime,
            )
        return {
            "schema_version": 1,
            **self.meter.snapshot(),
            "runner_process": cpu,
            "event_loop_lag": lag,
            "first_verification_submit_elapsed_s": self.first_submit_s,
            "engine": {
                "samples": self.engine_samples,
                "errors": list(self.engine_errors),
                "first": dict(self.engine_first),
                "last": dict(self.engine_last),
                "observed_max": dict(self.engine_max),
            },
            "scope": "runner CPU/IO only; async wall waits overlap; engine counters include warmup baseline",
        }


def grader_timeline(path, official_started_at_utc, target_correct, cost):
    """Oracle service versus idle gaps, independently of summed client waits."""
    from datetime import datetime

    if not path.exists():
        return {"completed_queries": 0, "target_service_floor_s": target_correct * cost}
    rows, errors = [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if all(k in row for k in ("picked_at", "answered_at", "verdict")):
                rows.append(row)
            else:
                errors.append("Incomplete audit row")
        except ValueError:
            errors.append("Invalid or interrupted audit JSON row")
    if not rows:
        return {
            "completed_queries": 0,
            "target_service_floor_s": target_correct * cost,
            "audit_errors": errors,
        }
    ts = lambda value: datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    ordered = sorted(rows, key=lambda r: ts(r["picked_at"]))
    service = sum(max(0.0, ts(r["answered_at"]) - ts(r["picked_at"])) for r in ordered)
    span = ts(ordered[-1]["answered_at"]) - ts(ordered[0]["picked_at"])
    return {
        "completed_queries": len(rows),
        "audit_errors": errors,
        "correct": sum(r["verdict"] is True for r in rows),
        "wrong": sum(r["verdict"] is False for r in rows),
        "target_service_floor_s": target_correct * cost,
        "actual_service_s": service,
        "idle_between_queries_s": max(0.0, span - service),
        "first_pick_elapsed_s": (
            ts(ordered[0]["picked_at"]) - ts(official_started_at_utc)
            if official_started_at_utc
            else None
        ),
        "scope": "completed oracle jobs; queued/cancelled client jobs may still incur service",
    }
