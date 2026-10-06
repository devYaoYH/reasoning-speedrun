"""Offline checks for data fetching, profile fallback, plots and the bundled example."""

import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from reasoning_speedrun import fetch_data
from reasoning_speedrun.analysis import plot_attempt
from reasoning_speedrun.lib import aime
from reasoning_speedrun.lib.setup import resolve_profile

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/attempts/20261004T220514.018752Z"


def parquet_bytes(rows):
    sink = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(rows), sink)
    return sink.getvalue()


def synthetic_rows():
    return [
        {"problem_idx": i, "problem": f"Problem {i}.", "answer": i * 10 % 1000, "problem_type": ["x"]}
        for i in range(1, 31)
    ]


class FetchDataTests(unittest.TestCase):
    def manifest_for(self, content, rows):
        problems = fetch_data.normalize_rows(rows)
        prompts = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in problems)
        grader = "".join(
            json.dumps({"problem_idx": r["problem_idx"], "problem": r["problem"], "answer": str(r["answer"])}) + "\n"
            for r in problems
        )
        return {
            "source": "https://huggingface.co/datasets/Org/aime_2025",
            "revision": "r" * 40,
            "dataset_sha256": hashlib.sha256(prompts.encode()).hexdigest(),
            "grader_sha256": hashlib.sha256(grader.encode()).hexdigest(),
            "downloads": [{"url": "https://example.test/train.parquet", "sha256": hashlib.sha256(content).hexdigest()}],
        }

    def run_fetch(self, manifest, content):
        client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=content)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "source.json"
            path.write_text(json.dumps(manifest))
            with patch.object(fetch_data, "benchmark_paths", return_value=(None, None, path)):
                return fetch_data.fetch(2025, client)

    def test_verified_download_reproduces_pinned_files(self):
        content = parquet_bytes(synthetic_rows())
        problems, prompts, grader = self.run_fetch(self.manifest_for(content, synthetic_rows()), content)
        self.assertEqual(len(problems), 30)
        self.assertNotIn('"answer": "10"', prompts)
        self.assertIn('"answer": "10"', grader)

    def test_tampered_download_or_output_is_rejected(self):
        content = parquet_bytes(synthetic_rows())
        manifest = self.manifest_for(content, synthetic_rows())
        with self.assertRaisesRegex(ValueError, "pinned hash"):
            self.run_fetch(manifest, content + b"x")
        manifest["downloads"][0]["sha256"] = None
        manifest["dataset_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "prompt file differs"):
            self.run_fetch(manifest, content)

    def test_missing_data_names_the_fetch_command(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"SPEEDRUN_DATA": tmp}):
            with self.assertRaisesRegex(FileNotFoundError, "fetch_data"):
                aime.dataset_provenance(2025)

    def test_manifests_pin_every_year(self):
        for year in aime.YEARS:
            manifest = json.loads(aime.benchmark_paths(year)[2].read_text())
            for key in ("source", "revision", "dataset_sha256", "grader_sha256", "license"):
                self.assertTrue(manifest.get(key), (year, key))


class ProfileTests(unittest.TestCase):
    def test_bundled_profile_points_at_models_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(
                models_dir=str(Path(tmp) / "models"),
                model="r0b0tlab/VibeThinker-3B-NVFP4",
                model_profile="vllm-flashinfer.yaml",
            )
            out = Path(tmp) / "out"
            out.mkdir()
            path = resolve_profile(args, out)
            profile = yaml.safe_load(path.read_text())
            self.assertEqual(profile["model"], str(Path(tmp) / "models" / args.model))
            self.assertEqual(profile["max-model-len"], 65536)

    def test_user_profile_wins_and_unknown_profile_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            models = Path(tmp) / "models"
            (models / "a/b").mkdir(parents=True)
            (models / "a/b/custom.yaml").write_text("model: /w\n")
            args = SimpleNamespace(models_dir=str(models), model="a/b", model_profile="custom.yaml")
            self.assertEqual(resolve_profile(args, Path(tmp)), models / "a/b/custom.yaml")
            args.model_profile = "missing.yaml"
            with self.assertRaises(FileNotFoundError):
                resolve_profile(args, Path(tmp))


class ExampleAttemptTests(unittest.TestCase):
    def test_example_is_self_describing_and_gold_free(self):
        self.assertTrue((EXAMPLE / "summary.json").is_file())
        questions = json.loads((EXAMPLE / "questions.json").read_text())
        self.assertTrue(all(set(q) == {"problem_idx", "problem"} for q in questions))

    def test_plot_renders_from_saved_telemetry(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = plot_attempt.render(EXAMPLE, Path(tmp))
            self.assertEqual(result["verified_correct"], 18)
            self.assertAlmostEqual(result["time_to_target_s"], 62.118150707974564)
            self.assertTrue((Path(tmp) / "attempt-timeline.png").stat().st_size > 10_000)


if __name__ == "__main__":
    unittest.main()
