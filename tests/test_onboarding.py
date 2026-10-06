"""What a newcomer hits first: clear errors, runnable docs."""

import io
import json
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import re
import shlex
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from qed import cli, fetch_data
from qed.lib import aime
from qed.lib.gpu import assert_gpu_idle

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "src/qed/examples"


def run_cli(argv, **env):
    out, err = io.StringIO(), io.StringIO()
    with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"QED_HOME": tmp, "QED_DATA": tmp, **env}):
        with redirect_stdout(out), redirect_stderr(err):
            try:
                cli.main(argv)
                code = 0
            except SystemExit as exc:
                code = exc.code
        attempts = sorted(Path(tmp, "attempts").glob("*")) if Path(tmp, "attempts").exists() else []
        summaries = [json.loads((a / "summary.json").read_text()) for a in attempts if (a / "summary.json").exists()]
    return code, out.getvalue(), err.getvalue(), summaries


class FriendlyErrorTests(unittest.TestCase):
    def test_missing_data_is_one_line_that_names_both_ways_forward(self):
        code, _, err, _ = run_cli([])
        self.assertEqual(code, 1)
        self.assertNotIn("Traceback", err)
        self.assertEqual(err.count("qed: error:"), 1)
        self.assertIn("qed fetch-data", err)
        self.assertIn("--simulate", err)
        self.assertIn("synthetic_grader.yaml", err)
        self.assertIn("QED_DEBUG", err)

    def test_debug_flag_restores_the_traceback(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"QED_HOME": tmp, "QED_DATA": tmp, "QED_DEBUG": "1"}):
            with self.assertRaises(FileNotFoundError):
                cli.main([])

    def test_no_gpu_driver_is_explained_and_not_masked_by_later_failures(self):
        import pynvml

        argv = ["--grader-config", str(EXAMPLES / "integer_grader.yaml"),
                "--system-prompt-file", str(EXAMPLES / "integer_prompt.txt"), "--parallelism", "2", "--target-correct", "2"]
        with patch.object(pynvml, "nvmlInit", side_effect=pynvml.NVMLError_LibraryNotFound()):
            code, _, err, summaries = run_cli(argv)
        self.assertEqual(code, 1)
        self.assertIn("No usable NVIDIA GPU driver", err)
        self.assertIn("--simulate", err)
        self.assertNotIn("Unsupported AIME year", err)
        self.assertEqual(summaries[0]["status"], "failed")
        self.assertIn("NVIDIA", summaries[0]["error"])

    def test_assert_gpu_idle_message(self):
        import pynvml

        with patch.object(pynvml, "nvmlInit", side_effect=pynvml.NVMLError_LibraryNotFound()):
            with self.assertRaisesRegex(RuntimeError, "--simulate"):
                assert_gpu_idle(0)

    def test_missing_vllm_binary_is_explained(self):
        from qed.lib.setup import prepare_inference
        import asyncio

        args = SimpleNamespace(simulate=False, reuse_server=False, gpu_device=0, vllm_binary="/no/such/vllm")
        with patch("qed.lib.setup.assert_gpu_idle"):
            with self.assertRaisesRegex(RuntimeError, "vLLM executable not found.*--simulate"):
                asyncio.run(prepare_inference(args, None, None, None, Path("."), {}, None, {}))

    def test_download_failure_names_the_host_and_writes_nothing(self):
        err = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"QED_DATA": tmp}), \
                patch.object(fetch_data, "fetch", side_effect=httpx.ConnectError("no route")), redirect_stderr(err):
            with self.assertRaises(SystemExit) as raised:
                fetch_data.main(["--year", "2025"])
            self.assertEqual(list(Path(tmp).iterdir()), [])
        self.assertIn("huggingface.co", str(raised.exception))
        self.assertIn("nothing was written", str(raised.exception))

    def test_viewer_without_attempts_points_at_the_bundled_example(self):
        from qed.viewer import server

        err = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"QED_HOME": tmp}), redirect_stderr(err):
            with self.assertRaises(SystemExit):
                server.main([])
        self.assertIn("examples/attempts", err.getvalue())
        self.assertIn("qed view", err.getvalue())

    def test_data_missing_message_for_the_simulator(self):
        from qed.sim import answer_key
        from qed.sim.config import SimConfig

        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"QED_DATA": tmp}):
            args = cli.parse_args(["--simulate"])
            with self.assertRaisesRegex(FileNotFoundError, "synthetic_grader.yaml"):
                answer_key(args, SimConfig.from_dict(args.simulation))


class DocsAreRunnableTests(unittest.TestCase):
    """Every `qed ...` command in the docs must at least parse against the real CLI."""

    def commands(self):
        for name in ("README.md", "docs/usage.md"):
            text = (ROOT / name).read_text()
            for block in re.findall(r"```bash\n(.*?)```", text, re.S):
                joined = re.sub(r"\\\n\s*", " ", block)
                for line in joined.splitlines():
                    line = line.split("  #")[0].strip()
                    if line.startswith("qed ") or line == "qed":
                        yield name, line

    def test_documented_commands_parse(self):
        from qed.extensions.naive.cli import parse_args as naive
        from qed.extensions.v1_6.cli import parse_args as v16
        from qed.sim import describe

        seen = 0
        previous = Path.cwd()
        os.chdir(ROOT)
        try:
            for name, line in self.commands():
                words = shlex.split(line)[1:]
                if "..." in words or any("<" in w for w in words):
                    continue  # placeholders
                seen += 1
                if words and words[0] == "sim-config":
                    with redirect_stdout(io.StringIO()):
                        describe.main(words[1:])
                    continue
                if words and words[0] in ("view", "fetch-data"):
                    continue
                version = words[words.index("--version") + 1] if "--version" in words else "v1"
                if "--version" in words:
                    i = words.index("--version")
                    words = words[:i] + words[i + 2 :]
                parse = {"v1": cli.parse_args, "naive": naive, "v1.6": v16}[version]
                try:
                    parse(words)
                except SystemExit as exc:  # --help exits 0; parser.error exits 2
                    if exc.code == 0:
                        continue
                    self.fail(f"{name}: `{line}` does not parse (exit {exc.code})")
        finally:
            os.chdir(previous)
        self.assertGreater(seen, 6)

    def test_files_the_docs_point_at_exist(self):
        for name in ("README.md", "docs/usage.md", "docs/README.md", "CONTRIBUTING.md"):
            base = (ROOT / name).parent
            for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", (ROOT / name).read_text()):
                if target.startswith(("http", "mailto:")):
                    continue
                self.assertTrue((base / target).exists(), f"{name} links to missing {target}")
        for path in re.findall(r"`((?:src/qed|examples|docs)/[^`\s]+?)`", (ROOT / "README.md").read_text()):
            self.assertTrue((ROOT / path.rstrip("/")).exists(), f"README mentions missing {path}")


if __name__ == "__main__":
    unittest.main()
