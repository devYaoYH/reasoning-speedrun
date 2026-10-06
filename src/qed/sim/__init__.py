"""``qed --simulate``: a mocked inference backend with configurable serving behavior.

The runner, scheduler, extraction, grader and traces are the real ones; only the
vLLM server (and the GPU) is replaced, in process, by ``Simulation``. See
``docs/usage.md#simulation`` for the knobs and what the model does and does not capture.
"""

import json
from pathlib import Path

import httpx
import yaml

from qed.lib import aime
from qed.lib.common import PACKAGE, workdir
from qed.sim.config import SimConfig, expected
from qed.sim.engine import Engine
from qed.sim.gpu import SimulatedGPUSampler
from qed.sim.transport import SimulatedTransport


class Key(dict):
    """Problem text -> integer answer, remembering each question's problem_idx for reports."""

    indices = {}


def read_key(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    key = Key()
    key.indices = {}
    for row in rows:
        text, answer = row.get("problem"), row.get("answer")
        if isinstance(text, str) and str(answer).strip().lstrip("-").isdigit():
            key[text.strip()] = int(answer)
            if row.get("problem_idx") is not None:
                key.indices[text.strip()] = row["problem_idx"]
    if not key:
        raise ValueError(f"{path}: no rows with a problem statement and an integer answer")
    return key


def answer_key(args, config):
    """Problem text -> integer answer, for the simulated model only (the solver stays gold-free)."""
    if config.behavior.answers:
        return read_key(Path(config.behavior.answers).expanduser())
    if args.grader_config:
        spec = yaml.safe_load(Path(args.grader_config).expanduser().read_text())["dataset"]
        source = Path(spec["source"]).expanduser()
        return read_key(source if source.is_absolute() else PACKAGE / "grader" / source)
    if args.dataset_manifest:
        manifest = Path(args.dataset_manifest).expanduser()
        manifest = manifest if manifest.is_absolute() else workdir() / manifest
        return read_key(workdir() / json.loads(manifest.read_text())["grader_path"])
    if args.reuse_grader:
        raise ValueError("--simulate with --reuse-grader needs a key: set behavior.answers (--sim behavior.answers=FILE)")
    path = aime.benchmark_paths(args.benchmark_year)[1]
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing: the simulated model needs an answer key. Run "
            f"`qed fetch-data --year {args.benchmark_year}`, or use your own questions "
            "(--grader-config) or --sim behavior.answers=FILE"
        )
    return read_key(path)


class Simulation:
    def __init__(self, config, engine, args):
        self.config, self.engine, self.port = config, engine, args.vllm_port

    @classmethod
    def create(cls, args, run_config, profile):
        config = SimConfig.from_dict(args.simulation)
        launch = profile or {}
        generation = launch.get("override-generation-config") or {}
        if isinstance(generation, str):
            generation = json.loads(generation)
        engine = Engine(
            config,
            args.model,
            answer_key(args, config),
            max_model_len=config.max_model_len or int(launch.get("max-model-len") or 65536),
            max_new_tokens=generation.get("max_new_tokens"),
            gpu_memory_utilization=config.gpu_memory_utilization or launch.get("gpu-memory-utilization") or 0.9,
        )
        # Recorded on args too: the runtime re-merges vars(args) into config.json.
        args.simulated = run_config["simulated"] = True
        args.simulation = run_config["simulation"] = {
            **config.to_dict(),
            "resolved": {
                "max_model_len": engine.max_model_len,
                "max_new_tokens": engine.max_new_tokens,
                "gpu_memory_utilization": engine.gpu_memory_utilization,
                "kv_capacity_tokens": engine.kv_capacity_tokens,
                "difficulty_tiers": expected(config.behavior)["tiers"],
            },
        }
        return cls(config, engine, args)

    def client_kwargs(self, limits=None):
        fallback = httpx.AsyncHTTPTransport(limits=limits or httpx.Limits())
        return {"transport": SimulatedTransport(self.engine, self.port, fallback)}

    def gpu_sampler(self, path, interval):
        return SimulatedGPUSampler(path, self.engine, interval)

    def report(self):
        return self.engine.report()
