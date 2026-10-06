"""Simulation knobs: validated, YAML/JSON friendly, overridable with ``--sim KEY=VALUE``."""

from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path

import yaml


def number(value):
    """YAML reads ``1e9`` as a string; accept numeric-looking text for numeric knobs."""
    if isinstance(value, str):
        for convert in (int, float):
            try:
                return convert(value)
            except ValueError:
                pass
    return value


@dataclass
class Behavior:
    """How the simulated *model* behaves: trajectory length, accuracy, when answers appear."""

    seed: int = 0
    p_correct: float = 0.65  # chance a sample's final answer is the answer key's
    reasoning_median_tokens: int = 6000  # lognormal length of the reasoning
    reasoning_sigma: float = 0.8
    reasoning_min_tokens: int = 200
    answer_at: list = field(default_factory=lambda: [0.45, 0.95])  # first answer, as a fraction of reasoning
    p_wrong_first: float = 0.15  # chance of a wrong tentative answer before the final one
    final_tokens: int = 40  # tokens of the final (non-reasoning) response
    answers: str | None = None  # JSONL of problem/answer; default: the run's own answer key


@dataclass
class SimConfig:
    """Coarse serving model: static per-request rates, a sequence cap, a prefix cache."""

    decode_tps: float = 100.0  # decode tokens/s of each running request (load independent)
    prefill_tps: float = 8000.0  # uncached prompt tokens/s of each request
    max_num_seqs: int = 256  # running sequences; further requests wait FIFO
    stream_interval_tokens: int = 8  # tokens per streamed chunk
    prefix_cache: bool = True
    block_tokens: int = 16  # prefix-cache block size
    max_model_len: int | None = None  # default: the launch profile's, else 65536
    gpu_memory_utilization: float | None = None  # default: the launch profile's, else 0.9
    vram_total_mib: float = 81920.0
    weights_mib: float = 2400.0
    kv_bytes_per_token: int = 36864  # 2 * layers * kv_heads * head_dim * dtype bytes
    behavior: Behavior = field(default_factory=Behavior)

    def validate(self):
        b = self.behavior
        positive = {
            "decode_tps": self.decode_tps,
            "prefill_tps": self.prefill_tps,
            "max_num_seqs": self.max_num_seqs,
            "stream_interval_tokens": self.stream_interval_tokens,
            "block_tokens": self.block_tokens,
            "vram_total_mib": self.vram_total_mib,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "behavior.reasoning_median_tokens": b.reasoning_median_tokens,
            "behavior.reasoning_min_tokens": b.reasoning_min_tokens,
            "behavior.final_tokens": b.final_tokens,
        }
        bad = [k for k, v in positive.items() if not isinstance(v, (int, float)) or v <= 0]
        if bad:
            raise ValueError("Simulation values must be positive: " + ", ".join(bad))
        if self.weights_mib < 0 or b.reasoning_sigma < 0:
            raise ValueError("weights_mib and behavior.reasoning_sigma must be nonnegative")
        for name, value in (("behavior.p_correct", b.p_correct), ("behavior.p_wrong_first", b.p_wrong_first)):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0, 1]")
        lo, hi = (b.answer_at + [None, None])[:2] if len(b.answer_at) == 2 else (None, None)
        if lo is None or not 0 <= lo <= hi <= 1:
            raise ValueError("behavior.answer_at must be [low, high] with 0 <= low <= high <= 1")
        if self.max_model_len is not None and self.max_model_len < 2:
            raise ValueError("max_model_len must be at least 2")
        if self.gpu_memory_utilization is not None and not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("gpu_memory_utilization must lie in (0, 1]")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, value):
        value = dict(value or {})
        behavior = value.pop("behavior", {}) or {}
        known = {f.name for f in fields(cls)} - {"behavior"}
        unknown = sorted(set(value) - known)
        bad = sorted(set(behavior) - {f.name for f in fields(Behavior)})
        if unknown or bad:
            raise ValueError(
                "Unknown simulation keys: "
                + ", ".join(unknown + [f"behavior.{k}" for k in bad])
                + ". Valid: "
                + ", ".join(sorted(known) + [f"behavior.{f.name}" for f in fields(Behavior)])
            )
        text = {"answers"}
        value = {k: number(v) for k, v in value.items()}
        behavior = {k: v if k in text else number(v) for k, v in behavior.items()}
        return cls(**value, behavior=Behavior(**behavior)).validate()

    @classmethod
    def from_sources(cls, path=None, overrides=()):
        """Defaults < config file < ``KEY=VALUE`` overrides (dotted for behavior.*)."""
        value = {}
        if path:
            text = Path(path).expanduser().read_text()
            value = (json.loads(text) if str(path).endswith(".json") else yaml.safe_load(text)) or {}
            if not isinstance(value, dict):
                raise ValueError("Simulation config must be a mapping")
        value = {**value, "behavior": dict(value.get("behavior") or {})}
        for item in overrides:
            key, sep, raw = item.partition("=")
            if not sep or not key.strip():
                raise ValueError(f"Simulation override must be KEY=VALUE: {item!r}")
            target = value
            parts = key.strip().split(".")
            for part in parts[:-1]:
                if part != "behavior":
                    raise ValueError(f"Unknown simulation key: {key}")
                target = target["behavior"]
            target[parts[-1]] = number(yaml.safe_load(raw))
        return cls.from_dict(value)
