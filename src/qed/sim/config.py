"""Simulation knobs: validated, YAML/JSON friendly, overridable with ``--sim KEY=VALUE``."""

from dataclasses import asdict, dataclass, field, fields, replace
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
    # Per-question difficulty sets WHEN answers appear (length, answer_at), not accuracy:
    # "mixed" (easy/medium/hard), "uniform" (every question alike), or a list of tiers.
    difficulty: str | list | None = "mixed"

    def validate(self):
        positive = {"reasoning_median_tokens": self.reasoning_median_tokens,
                    "reasoning_min_tokens": self.reasoning_min_tokens, "final_tokens": self.final_tokens}
        bad = [f"behavior.{k}" for k, v in positive.items() if not isinstance(v, (int, float)) or v <= 0]
        if bad:
            raise ValueError("Simulation values must be positive: " + ", ".join(bad))
        if not isinstance(self.reasoning_sigma, (int, float)) or self.reasoning_sigma < 0:
            raise ValueError("behavior.reasoning_sigma must be nonnegative")
        for name in ("p_correct", "p_wrong_first"):
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"behavior.{name} must lie in [0, 1]")
        answer_at = self.answer_at
        if not (isinstance(answer_at, list) and len(answer_at) == 2 and all(isinstance(x, (int, float)) for x in answer_at)
                and 0 <= answer_at[0] <= answer_at[1] <= 1):
            raise ValueError("behavior.answer_at must be [low, high] with 0 <= low <= high <= 1")
        return self


# Knobs a difficulty tier may override; anything it leaves out is inherited from `behavior`.
TIER_KNOBS = ("p_correct", "reasoning_median_tokens", "reasoning_sigma", "answer_at", "p_wrong_first")

PRESETS = {
    # Easy/medium/hard in the ratio 7:5:3 (14/10/6 of 30 questions). Difficulty controls WHEN
    # answers appear, not whether they are right: accuracy stays `p_correct` in every tier.
    # A preset is relative to the flat knobs: `length` scales reasoning length (calibrated so
    # the weighted mean equals the flat length) and `answer_shift` moves the `answer_at`
    # window earlier (easy) or later (hard).
    "mixed": [
        {"name": "easy", "weight": 7, "length": 0.433, "answer_shift": -0.15},
        {"name": "medium", "weight": 5, "length": 1.083, "answer_shift": 0.0},
        {"name": "hard", "weight": 3, "length": 2.167, "answer_shift": 0.15},
    ],
}


def calibrated(behavior, preset):
    """Absolute tiers for a relative preset; accuracy is untouched."""
    total = sum(row["weight"] for row in preset)
    scale = sum(row["weight"] / total * row["length"] for row in preset)
    low, high = behavior.answer_at
    tiers_ = []
    for row in preset:
        shift = row["answer_shift"]
        window = [min(1.0, max(0.0, low + shift)), min(1.0, max(0.0, high + shift))]
        tiers_.append({
            "name": row["name"],
            "weight": row["weight"],
            "reasoning_median_tokens": max(1, round(behavior.reasoning_median_tokens * row["length"] / scale)),
            "answer_at": [round(window[0], 4), round(max(window[0], window[1]), 4)],
        })
    return tiers_


def tiers(behavior):
    """Resolved difficulty tiers: ``[{"name", "weight" (normalized), "behavior"}]``.

    ``uniform`` (or unset) is a single tier equal to the flat knobs, so a configuration
    without ``difficulty`` behaves exactly as it always did.
    """
    spec = behavior.difficulty
    if spec in (None, "uniform"):
        spec = [{"name": "uniform", "weight": 1}]
    elif isinstance(spec, str):
        if spec not in PRESETS:
            raise ValueError(f"behavior.difficulty must be uniform, {', '.join(PRESETS)} or a list of tiers (got {spec!r})")
        spec = calibrated(behavior, PRESETS[spec])
    if not isinstance(spec, list) or not spec:
        raise ValueError("behavior.difficulty must be a name or a nonempty list of tiers")
    resolved = []
    for index, tier in enumerate(spec):
        if not isinstance(tier, dict):
            raise ValueError("Each difficulty tier must be a mapping")
        unknown = sorted(set(tier) - {"name", "weight", *TIER_KNOBS})
        if unknown:
            raise ValueError(f"Unknown difficulty tier keys: {', '.join(unknown)}. Valid: name, weight, {', '.join(TIER_KNOBS)}")
        weight = number(tier.get("weight"))
        if not isinstance(weight, (int, float)) or weight <= 0:
            raise ValueError("Each difficulty tier needs a positive weight")
        overrides = {k: (v if k == "answer_at" else number(v)) for k, v in tier.items() if k in TIER_KNOBS}
        flat = replace(behavior, difficulty=None, **overrides)
        flat.validate()
        resolved.append({"name": str(tier.get("name", f"tier{index + 1}")), "weight": weight, "behavior": flat})
    total = sum(r["weight"] for r in resolved)
    for r in resolved:
        r["weight"] /= total
    return resolved


def expected(behavior):
    """Headline statistics implied by the tiers: what the knobs add up to."""
    resolved = tiers(behavior)
    import math

    def mean_len(b):
        return b.reasoning_median_tokens * math.exp(b.reasoning_sigma**2 / 2)

    return {
        "tiers": [
            {"name": r["name"], "weight": round(r["weight"], 4),
             **{k: (list(v) if isinstance(v, list) else v) for k in TIER_KNOBS for v in [getattr(r["behavior"], k)]},
             # Typical token position of the first answer: median length times the window midpoint.
             "typical_first_answer_tokens": round(r["behavior"].reasoning_median_tokens * sum(r["behavior"].answer_at) / 2)}
            for r in resolved
        ],
        "mean_p_correct": sum(r["weight"] * r["behavior"].p_correct for r in resolved),
        "mean_reasoning_tokens": sum(r["weight"] * mean_len(r["behavior"]) for r in resolved),
        # Samples of one question share its tier, so they are correlated: this is the chance
        # that four samples of a question all end wrong (independent samples: (1-p)^4).
        "p_four_samples_all_wrong": sum(r["weight"] * (1 - r["behavior"].p_correct) ** 4 for r in resolved),
        # Share of questions with at least one correct sample among four: the ceiling on how
        # many questions a four-sample policy can solve (target reachability).
        "p_question_solvable_in_four_samples": sum(r["weight"] * (1 - (1 - r["behavior"].p_correct) ** 4) for r in resolved),
    }


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
        positive = {
            "decode_tps": self.decode_tps,
            "prefill_tps": self.prefill_tps,
            "max_num_seqs": self.max_num_seqs,
            "stream_interval_tokens": self.stream_interval_tokens,
            "block_tokens": self.block_tokens,
            "vram_total_mib": self.vram_total_mib,
            "kv_bytes_per_token": self.kv_bytes_per_token,
        }
        bad = [k for k, v in positive.items() if not isinstance(v, (int, float)) or v <= 0]
        if bad:
            raise ValueError("Simulation values must be positive: " + ", ".join(bad))
        if not isinstance(self.weights_mib, (int, float)) or self.weights_mib < 0:
            raise ValueError("weights_mib must be nonnegative")
        self.behavior.validate()
        tiers(self.behavior)  # validates difficulty and every tier
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
