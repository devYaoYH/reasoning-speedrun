"""Select versioned AIME datasets and record the exact inputs used by a run."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from reasoning_speedrun.lib.common import PACKAGE

YEARS = (2024, 2025, 2026)
ROLES = ("prewarming", "development", "generalization")


def default_benchmark_role(year):
    return {2024: "prewarming", 2025: "development", 2026: "generalization"}[year]


def add_dataset_args(parser):
    parser.add_argument(
        "--benchmark-year",
        type=int,
        choices=YEARS,
        default=2025,
        help="AIME year; default 2025 keeps existing development runs unchanged",
    )
    parser.add_argument(
        "--benchmark-role",
        choices=ROLES,
        help="Defaults to prewarming for 2024, development for 2025, generalization for 2026",
    )


def data_dir():
    """Fetched benchmark files (licensed upstream, never bundled): ``$SPEEDRUN_DATA`` or the user cache."""
    return Path(
        os.environ.get("SPEEDRUN_DATA") or Path.home() / ".cache" / "reasoning_speedrun"
    )


def benchmark_paths(year=2025):
    """Return (prompt file, grader key file, packaged provenance manifest)."""
    if year not in YEARS:
        raise ValueError(f"Unsupported AIME year: {year}")
    return (
        data_dir() / f"aime_{year}_problems.jsonl",
        data_dir() / "grader" / f"aime_{year}.jsonl",
        PACKAGE / "data" / ("source.json" if year == 2025 else f"source_{year}.json"),
    )


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing; fetch the pinned dataset with "
            "`reasoning-speedrun fetch-data --year YEAR` "
            "(requires `pip install reasoning-speedrun[data]`)"
        )
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if [r["problem_idx"] for r in rows] != list(range(1, 31)):
        raise ValueError("Expected exactly 30 ordered AIME problem indices")
    if any(
        not isinstance(r.get("problem"), str) or not r["problem"].strip() for r in rows
    ):
        raise ValueError("Empty AIME problem statement")
    return rows


def load_questions(indices=None, year=2025):
    """Return solver prompts with gold answers stripped."""
    prompts, _, _ = benchmark_paths(year)
    problems = [
        {"problem_idx": r["problem_idx"], "problem": r["problem"]}
        for r in read_rows(prompts)
    ]
    if indices:
        requested = set(indices)
        if not requested <= {p["problem_idx"] for p in problems}:
            raise ValueError("Unknown question index")
        problems = [p for p in problems if p["problem_idx"] in requested]
    return problems


def dataset_provenance(year=2025, role=None):
    """Fail before services start if the prompt file and grader key disagree."""
    prompts, grader, manifest_path = benchmark_paths(year)
    rows, gold = read_rows(prompts), read_rows(grader)
    for prompt, key in zip(rows, gold):
        answer = prompt.get("answer")
        if type(answer) is not int or not 0 <= answer <= 999:
            raise ValueError("Expected an AIME integer answer between 0 and 999")
        if prompt["problem"] != key["problem"] or str(answer) != str(key.get("answer")):
            raise ValueError(
                f'Prompt/grader mismatch for AIME {year} #{prompt["problem_idx"]}'
            )
    manifest = json.loads(manifest_path.read_text())
    prompt_hash, grader_hash = sha256(prompts), sha256(grader)
    if manifest["dataset_sha256"] != prompt_hash:
        raise ValueError("Benchmark prompt hash does not match its source manifest")
    if manifest.get("grader_sha256", grader_hash) != grader_hash:
        raise ValueError("Benchmark grader hash does not match its source manifest")
    evidence = {
        "id": f"aime_{year}",
        "year": year,
        "role": role or default_benchmark_role(year),
        "source": manifest["source"],
        "revision": manifest["revision"],
        "split": manifest["split"],
        "rows": len(rows),
        "prompt_path": prompts.name,
        "prompt_sha256": prompt_hash,
        "grader_path": f"grader/{grader.name}",
        "grader_sha256": grader_hash,
        "inferred_from_legacy_runner": False,
    }
    if manifest.get("sources"):
        evidence["sources"] = manifest["sources"]
    return evidence


def recorded_dataset(config):
    """Historical runners were fixed to 2025; never invent historical hashes."""
    if config.get("dataset_provenance"):
        evidence = config["dataset_provenance"]
        benchmark_paths(evidence["year"])
        if (
            evidence["id"] != f"aime_{evidence['year']}"
            or config.get("benchmark_year", evidence["year"]) != evidence["year"]
        ):
            raise ValueError(
                "Recorded benchmark year disagrees with dataset provenance"
            )
        return evidence
    year = config.get("benchmark_year", 2025)
    benchmark_paths(year)  # Validate before using a saved year to select a file.
    return {
        "id": f"aime_{year}",
        "year": year,
        "role": config.get("benchmark_role"),
        "source": f"https://huggingface.co/datasets/MathArena/aime_{year}",
        "revision": None,
        "split": "train",
        "rows": 30,
        "prompt_path": f"aime_{year}_problems.jsonl",
        "prompt_sha256": None,
        "grader_path": f"grader/aime_{year}.jsonl",
        "grader_sha256": None,
        "inferred_from_legacy_runner": "benchmark_year" not in config,
    }
