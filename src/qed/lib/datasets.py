"""Pinned arbitrary-size mathematical datasets; solver inputs never include gold."""

import hashlib
import json
from pathlib import Path

from qed.lib.common import workdir
from qed.lib import aime


def default_benchmark_role(year):
    return aime.default_benchmark_role(year) if year is not None else "generalization"


def add_dataset_args(parser):
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--benchmark-year",
        type=int,
        choices=aime.YEARS,
        help="AIME year (default 2025 unless a dataset manifest is selected)",
    )
    selection.add_argument(
        "--dataset-manifest", help="Pinned JSON manifest for a mathematical dataset"
    )
    selection.add_argument(
        "--grader-config",
        help="Grader YAML dataset configuration; questions come from its API",
    )
    selection.add_argument(
        "--reuse-grader",
        action="store_true",
        help="Use an already running, fresh v2 grader on --grader-port; it supplies questions",
    )
    parser.add_argument("--benchmark-role", choices=aime.ROLES)


def manifest_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (workdir() / path).resolve()


def benchmark_paths(year=2025, manifest=None):
    if manifest is None:
        return aime.benchmark_paths(year)
    path = manifest_path(manifest)
    info = json.loads(path.read_text())
    # Manifest paths are relative to the working directory unless absolute.
    return (workdir() / info["prompt_path"], workdir() / info["grader_path"], path)


def read_rows(path, gold=False):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    indices = [row.get("problem_idx") for row in rows]
    if (
        not rows
        or any(type(i) is not int or i < 1 for i in indices)
        or indices != sorted(set(indices))
    ):
        raise ValueError("Expected nonempty, ordered, unique positive question indices")
    for row in rows:
        if not isinstance(row.get("problem"), str) or not row["problem"].strip():
            raise ValueError("Empty mathematical problem statement")
        if gold and (
            type(row.get("answer")) not in (str, int) or not str(row["answer"]).strip()
        ):
            raise ValueError("Expected a nonempty exact answer string or integer")
    return rows


def dataset_provenance(year=2025, role=None, manifest=None):
    if manifest is None:
        return aime.dataset_provenance(year, role)
    prompts, grader, path = benchmark_paths(year, manifest)
    info = json.loads(path.read_text())
    if (
        info.get("schema_version") != 1
        or not isinstance(info.get("id"), str)
        or not info["id"]
    ):
        raise ValueError("Invalid mathematical dataset manifest")
    rows, gold = read_rows(prompts), read_rows(grader, gold=True)
    if type(info.get("rows")) is not int or info["rows"] != len(rows):
        raise ValueError("Dataset row count disagrees with manifest")
    if [(r["problem_idx"], r["problem"]) for r in rows] != [
        (r["problem_idx"], r["problem"]) for r in gold
    ]:
        raise ValueError("Prompt/grader indices or statements disagree")
    prompt_hash, grader_hash = aime.sha256(prompts), aime.sha256(grader)
    if (
        info.get("prompt_sha256") != prompt_hash
        or info.get("grader_sha256") != grader_hash
    ):
        raise ValueError("Dataset hash does not match its source manifest")
    return {
        "id": info["id"],
        "year": None,
        "role": role or "generalization",
        "source": info["source"],
        "revision": info["revision"],
        "split": info["split"],
        "rows": len(rows),
        "prompt_path": info["prompt_path"],
        "prompt_sha256": prompt_hash,
        "grader_path": info["grader_path"],
        "grader_sha256": grader_hash,
        "inferred_from_legacy_runner": False,
        "manifest_path": (
            str(path.relative_to(workdir()))
            if path.is_relative_to(workdir())
            else str(path)
        ),
        "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "overlap_notes": info.get("overlap_notes", []),
    }


def load_questions(indices=None, year=2025, manifest=None):
    if manifest is None:
        return aime.load_questions(indices, year)
    prompts, _, _ = benchmark_paths(year, manifest)
    problems = [
        {"problem_idx": row["problem_idx"], "problem": row["problem"]}
        for row in read_rows(prompts)
    ]
    if indices is not None:
        requested = set(indices)
        if not requested or not requested <= {p["problem_idx"] for p in problems}:
            raise ValueError("Unknown or empty question subset")
        problems = [p for p in problems if p["problem_idx"] in requested]
    return problems


def recorded_dataset(config):
    if config.get("dataset_provenance"):
        return config["dataset_provenance"]
    return aime.recorded_dataset(config)
