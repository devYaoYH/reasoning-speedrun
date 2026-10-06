"""Offline checks of the naive policy: full fan-out, final answers only."""

import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from pathlib import Path
import tempfile
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
import io

import httpx

from reasoning_speedrun.extensions.naive.cli import RUNNER_ID, metadata, parse_args, prepare_problems
from reasoning_speedrun.extensions.naive.policy import final_answer, run_speedrun
from reasoning_speedrun.lib.metrics import AttemptProfiler
from reasoning_speedrun.lib.storage import AttemptArtifacts, DisabledGPUSampler
from tests.helpers import Stream, chunk, fake_provenance


def natural(content, reasoning=""):
    parts = []
    if reasoning:
        parts.append(chunk(reasoning, part="reasoning"))
    parts += [chunk(content), chunk("", finish="stop"), b"data: [DONE]\n\n"]
    return parts


class FinalAnswerTests(unittest.TestCase):
    def test_last_closed_integer_box_after_think(self):
        self.assertEqual(final_answer("so \\boxed{7} no wait \\boxed{070}")["answer"], 70)
        self.assertEqual(final_answer("<think>x \\boxed{9}</think> done \\boxed{12}")["answer"], 12)
        self.assertIsNone(final_answer("<think>still thinking \\boxed{9}"))
        self.assertIsNone(final_answer("answer is 70"))
        self.assertIsNone(final_answer("\\boxed{1000}"))
        self.assertIsNone(final_answer("\\boxed{12"))


class ConfigTests(unittest.TestCase):
    def test_defaults_and_forbidden_scheduling_flags(self):
        args = parse_args(["--target-correct", "2"])
        self.assertEqual(
            (args.rollouts, args.max_attempts_per_question, args.max_tokens, args.schedule),
            (4, 4, 16384, "full_fanout"),
        )
        self.assertTrue(args.no_continuation)
        self.assertEqual(parse_args(["--samples-per-question", "2"]).rollouts, 2)
        for flag in ("--parallelism", "--rollouts", "--schedule", "--max-rounds"):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args([flag, "3"])


class NaivePolicyTests(unittest.IsolatedAsyncioTestCase):
    async def run_policy(self, scripts, *, questions=2, options=(), verdict="70", gate=None):
        """scripts[(question, sample)] -> SSE chunks; returns (rows, requests, checks, streams)."""
        requests, checks, streams = [], [], {}

        async def handler(request):
            body = json.loads(request.content)
            if request.url.path == "/verify":
                checks.append((body["index"], body["candidate"]))
                await asyncio.sleep(0.005)
                return httpx.Response(200, json={"verdict": body["candidate"] == verdict})
            question = int(request.headers["X-Request-Id"].split("-q")[1].split("-r")[0])
            sample = int(request.headers["X-Request-Id"].rsplit("-r", 1)[1])
            requests.append((question, sample, body["max_tokens"]))
            if gate is not None:
                gate["started"] += 1
                if gate["started"] == gate["need"]:
                    gate["event"].set()
                await gate["event"].wait()
            stream = Stream(scripts[(question, sample)], delay=0.001)
            streams[(question, sample)] = stream
            return httpx.Response(200, stream=stream)

        args = parse_args(["--target-correct", "1", "--samples-per-question", "2", *options])
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            store = AttemptArtifacts(root, buffered=True)
            profiler = AttemptProfiler(root, enabled=False, artifacts=store)
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                problems = [{"problem_idx": i, "problem": f"Problem {i}."} for i in range(1, questions + 1)]
                args.parallelism = len(problems)
                rows = await run_speedrun(
                    problems, args, client, root, DisabledGPUSampler(), time.perf_counter(), profiler
                )
        return rows, requests, checks, streams

    async def test_boxes_in_reasoning_are_ignored_until_the_final_response(self):
        scripts = {
            (1, 1): natural("I could not finish.", reasoning="Maybe \\boxed{70}? Not sure.\n"),
            (1, 2): natural("I could not finish.", reasoning="Answer: 70\n"),
        }
        rows, requests, checks, _ = await self.run_policy(
            scripts, questions=1, options=["--target-correct", "1"]
        )
        self.assertEqual(checks, [])
        self.assertEqual(rows[0]["status"], "unsolved")
        self.assertEqual(sorted(r[:2] for r in requests), [(1, 1), (1, 2)])

    async def test_final_box_is_verified_only_after_natural_end_and_cancels_siblings(self):
        scripts = {
            (1, 1): natural("Therefore \\boxed{70}", reasoning="thinking \\boxed{99}\n"),
            (1, 2): natural("slow"),
        }
        rows, requests, checks, streams = await self.run_policy(scripts, questions=1)
        self.assertEqual(checks, [(1, "70")])
        self.assertEqual(rows[0]["status"], "solved")
        self.assertEqual(rows[0]["winner"]["kind"], "final_box")
        self.assertEqual(rows[0]["first_solved"]["candidate"], "70")
        self.assertTrue(all(s.closed for s in streams.values()))

    async def test_all_samples_of_all_questions_start_before_any_finishes(self):
        gate = {"started": 0, "need": 4, "event": asyncio.Event()}
        scripts = {(q, s): natural("\\boxed{5}") for q in (1, 2) for s in (1, 2)}
        scripts[(1, 1)] = natural("\\boxed{70}")
        rows, requests, checks, _ = await asyncio.wait_for(
            self.run_policy(scripts, gate=gate), timeout=10
        )
        self.assertEqual(gate["started"], 4)
        self.assertEqual({r[2] for r in requests}, {16384})
        self.assertEqual(len(requests), 4)

    async def test_length_capped_response_yields_no_candidate(self):
        capped = [chunk("\\boxed{70}"), chunk("", finish="length"), b"data: [DONE]\n\n"]
        scripts = {(1, 1): capped, (1, 2): capped}
        rows, _, checks, _ = await self.run_policy(scripts, questions=1)
        self.assertEqual(checks, [])
        self.assertEqual(rows[0]["status"], "unsolved")


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_lifecycle_records_naive_controls_and_valid_metadata(self):
        from reasoning_speedrun.lib import policy_runtime as runtime
        from reasoning_speedrun.lib.metadata import validate_metadata

        args = parse_args(["--questions", "1", "--target-correct", "1", "--samples-per-question", "2"])
        started = []

        async def handler(request):
            if request.url.path == "/verify":
                return httpx.Response(200, json={"verdict": True})
            started.append(json.loads(request.content)["max_tokens"])
            return httpx.Response(200, stream=Stream(natural("\\boxed{070}"), delay=0))

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        services = SimpleNamespace(close=AsyncMock())

        async def inference(*values):
            args.max_context_tokens = 65536

        async def warmup(*values):
            return dict(latency_s=0.0, batch_size=1, tokens_per_request=32)

        def local(a, config):
            config["dataset_provenance"] = fake_provenance(2025)
            return [dict(problem_idx=1, problem="test")]

        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            with (
                patch.dict(os.environ, {"SPEEDRUN_HOME": str(root)}),
                patch.object(runtime, "Services", return_value=services),
                patch.object(runtime, "ensure_free"),
                patch.object(runtime, "git_state", return_value=dict(git_commit=None, git_dirty=None)),
                patch.object(runtime, "local_dataset", side_effect=local),
                patch.object(
                    runtime,
                    "read_profile",
                    return_value=(root / "profile.yaml", "max-model-len: 65536", {"max-model-len": 65536}),
                ),
                patch.object(runtime, "prepare_inference", side_effect=inference),
                patch.object(
                    runtime, "prepare_grader",
                    new=AsyncMock(return_value=[dict(problem_idx=1, problem="test")]),
                ),
                patch.object(runtime, "warm_inference", side_effect=warmup),
                patch.object(runtime.httpx, "AsyncClient", return_value=client),
            ):
                output = await runtime.run_policy(
                    args,
                    policy=run_speedrun,
                    prepare_problems=prepare_problems,
                    verify=lambda root: "test-manifest",
                    runner_id=RUNNER_ID,
                    runner_module="reasoning_speedrun.extensions.naive",
                    metadata_builder=metadata,
                )
            summary = json.loads((output / "summary.json").read_text())
            config = json.loads((output / "config.json").read_text())
            meta = json.loads((output / "metadata.json").read_text())
            self.assertEqual(started, [16384, 16384])
            self.assertTrue(summary["target_reached"])
            self.assertEqual(summary["runner_id"], RUNNER_ID)
            self.assertEqual(summary["performance"]["continuation_requests"], 0)
            self.assertEqual(config["schedule"], "full_fanout")
            hp = meta["controls"]["hyperparameters"]
            self.assertEqual(
                (hp["schedule"], hp["max_attempts_per_question"], hp["continuation_enabled"], hp["answer_extraction"]),
                ("full_fanout", 2, False, "completed final box only"),
            )
            validate_metadata(meta, output.name)
            services.close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
