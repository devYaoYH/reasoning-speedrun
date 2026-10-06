"""Download revision-pinned MathArena AIME data into the user data directory.

    python -m reasoning_speedrun.fetch_data --year 2025

The benchmark files are licensed upstream (CC BY-NC-SA 4.0) and are therefore not
bundled with this package. The pinned revisions, Parquet hashes and output hashes
live in the packaged ``data/source*.json`` manifests; the downloaded bytes and the
generated files must match them. Files go to ``$SPEEDRUN_DATA`` or
``~/.cache/reasoning_speedrun``. Requires ``pip install reasoning-speedrun[data]``.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys

import httpx

from reasoning_speedrun.lib.aime import YEARS, benchmark_paths


def normalize_rows(rows):
    problems = []
    for row in rows:
        if type(row.get("problem_idx")) is not int or type(row.get("answer")) is not int:
            raise ValueError("AIME indices and answers must be integers")
        if not 0 <= row["answer"] <= 999:
            raise ValueError("AIME answers must lie between 0 and 999")
        if not isinstance(row.get("problem"), str) or not row["problem"].strip():
            raise ValueError("Empty AIME problem statement")
        problems.append(
            {key: row[key] for key in ("problem_idx", "problem", "answer")}
            | {"problem_type": row.get("problem_type", [])}
        )
    problems.sort(key=lambda row: row["problem_idx"])
    if [r["problem_idx"] for r in problems] != list(range(1, 31)):
        raise ValueError("Expected exactly 30 unique AIME problem indices")
    return problems


def combine_2024_parts(parts):
    """Validate each paper before mapping II's indices 1-15 to 16-30."""
    if len(parts) != 2:
        raise ValueError("Expected both AIME I and II 2024 papers")
    combined = []
    for offset, rows in zip((0, 15), parts):
        if any(type(r.get("problem_idx")) is not int for r in rows) or sorted(
            r["problem_idx"] for r in rows
        ) != list(range(1, 16)):
            raise ValueError("Expected 15 unique local indices in each AIME 2024 paper")
        combined.extend({**row, "problem_idx": row["problem_idx"] + offset} for row in rows)
    return normalize_rows(combined)


def parquet_downloads(client, manifest):
    """Pinned files, or the train Parquet files of the pinned revision when unlisted."""
    if manifest.get("downloads"):
        return manifest["downloads"]
    dataset = manifest["source"].removeprefix("https://huggingface.co/datasets/")
    api = f"https://huggingface.co/api/datasets/{dataset}/revision/{manifest['revision']}"
    info = client.get(api).raise_for_status().json()
    files = sorted(
        f["rfilename"]
        for f in info["siblings"]
        if f["rfilename"].startswith("data/train-") and f["rfilename"].endswith(".parquet")
    )
    if not files:
        raise ValueError("No train Parquet files in the pinned dataset")
    return [
        {"url": f"https://huggingface.co/datasets/{dataset}/resolve/{manifest['revision']}/{name}", "sha256": None}
        for name in files
    ]


def download_parquet(client, download):
    content = client.get(download["url"]).raise_for_status().content
    if download["sha256"] and hashlib.sha256(content).hexdigest() != download["sha256"]:
        raise ValueError(f"Downloaded file differs from the pinned hash: {download['url']}")
    import pyarrow.parquet as pq

    return pq.read_table(io.BytesIO(content)).to_pylist()


def fetch(year, client):
    _, _, manifest_path = benchmark_paths(year)
    manifest = json.loads(manifest_path.read_text())
    parts = [download_parquet(client, d) for d in parquet_downloads(client, manifest)]
    problems = combine_2024_parts(parts) if year == 2024 else normalize_rows(parts[0])
    prompt_text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in problems)
    grader_text = "".join(
        json.dumps(
            {**{k: r[k] for k in ("problem_idx", "problem")}, "answer": str(r["answer"])},
            ensure_ascii=False,
        )
        + "\n"
        for r in problems
    )
    if hashlib.sha256(prompt_text.encode()).hexdigest() != manifest["dataset_sha256"]:
        raise ValueError("Generated prompt file differs from the pinned manifest")
    if hashlib.sha256(grader_text.encode()).hexdigest() != manifest["grader_sha256"]:
        raise ValueError("Generated grader file differs from the pinned manifest")
    return problems, prompt_text, grader_text


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--year", type=int, choices=YEARS, action="append", help="Repeatable; default 2025")
    args = parser.parse_args(argv)
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        sys.exit("pyarrow is required: pip install 'reasoning-speedrun[data]'")
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        for year in args.year or [2025]:
            problems, prompt_text, grader_text = fetch(year, client)
            prompts, grader, _ = benchmark_paths(year)
            for path, content in ((prompts, prompt_text), (grader, grader_text)):
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".jsonl.tmp")
                tmp.write_text(content, encoding="utf-8")
                tmp.replace(path)
            print(f"Saved {len(problems)} AIME {year} problems to {prompts.parent}")


if __name__ == "__main__":
    main()
