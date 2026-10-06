"""Offline fresh-cap, exact-prefix, pool, error and cancellation checks."""

import asyncio
from contextlib import redirect_stdout, redirect_stderr
import io
import os
import json
from pathlib import Path
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock

import httpx
from reasoning_speedrun.extensions.v1_6.cli import parse_args, prepare_problems
from reasoning_speedrun.extensions.v1_6.policy import run_speedrun
from reasoning_speedrun.lib.metrics import AttemptProfiler
from reasoning_speedrun.lib.storage import AttemptArtifacts, DisabledGPUSampler
from tests.helpers import Stream, chunk


def capped(body, text="thinking"):
    prompt = body.get("prompt", [101, 102, 103])
    tokens = list(range(500, 500 + body["max_tokens"]))
    choice = dict(index=0, token_ids=tokens, finish_reason="length")
    if "prompt" in body:
        choice.update(text=text, prompt_token_ids=prompt)
    else:
        choice["delta"] = {"content": text}
    payload = dict(
        choices=[choice],
        prompt_token_ids=prompt,
        usage=dict(prompt_tokens=len(prompt), completion_tokens=len(tokens)),
    )
    return [("data: " + json.dumps(payload) + "\n\n").encode(), b"data: [DONE]\n\n"]


class PolicyTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, handler, *, count=1, changes=None):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            args = parse_args(
                ["--benchmark", "--target-correct", "1", "--question-timeout", "2"]
            )
            args.max_context_tokens = 100
            args.max_concurrent_requests = count
            vars(args).update(changes or {})
            artifacts = AttemptArtifacts(output, buffered=True)
            profiler = AttemptProfiler(output, enabled=False, artifacts=artifacts)
            problems = [
                dict(problem_idx=i, problem=str(i), prompt_tokens=3)
                for i in range(1, count + 1)
            ]
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                with redirect_stdout(io.StringIO()):
                    rows = await asyncio.wait_for(
                        run_speedrun(
                            problems,
                            args,
                            client,
                            output,
                            DisabledGPUSampler(),
                            time.perf_counter(),
                            profiler,
                        ),
                        3,
                    )
            allocation = artifacts.read_json(output / "allocation.json")
            self.assertFalse((output / "allocation.json").exists())
            artifacts.flush()
            return rows, allocation

    async def test_four_fresh_samples_each_continue_past_four_http_requests(self):
        requests = []

        async def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            return httpx.Response(200, stream=Stream(capped(body), delay=0))

        rows, allocation = await self.exercise(
            handler,
            changes={
                "first_pass_max_tokens": 2,
                "max_rollout_tokens": 13,
            },
        )
        self.assertEqual(len(requests), 8)
        self.assertEqual([b["max_tokens"] for b in requests], [2, 11] * 4)
        self.assertEqual(sum("messages" in b for b in requests), 4)
        self.assertEqual(allocation["questions"]["1"]["fresh"], 4)
        self.assertEqual(allocation["questions"]["1"]["requests"], 8)
        for sample in range(4):
            segment = requests[sample * 2 : (sample + 1) * 2]
            self.assertEqual([len(b.get("prompt", [])) for b in segment], [0, 5])
            self.assertEqual(segment[1]["prompt"], [101, 102, 103, 500, 501])
        self.assertEqual(
            [b["seed"] for b in requests[::2]], [20261008, 20261009, 20261010, 20261011]
        )
        self.assertEqual(len(set(b["seed"] for b in requests)), 8)
        self.assertTrue(
            all(r["trajectory_generated_tokens"] <= 13 for r in rows[0]["rollouts"])
        )
        self.assertEqual(rows[0]["status"], "unsolved")

    async def test_pass4_does_not_consume_continuation_allowance(self):
        requests = []

        async def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            return httpx.Response(200, stream=Stream(capped(body), delay=0.001))

        rows, allocation = await self.exercise(
            handler,
            changes={
                "max_concurrent_requests": 4,
                "first_pass_max_tokens": 2,
                "max_rollout_tokens": 6,
            },
        )
        self.assertEqual(sum("messages" in b for b in requests), 4)
        self.assertEqual(len(requests), 8)
        self.assertEqual(allocation["peak_active_requests"], 4)
        self.assertEqual(
            sorted(r["fresh_sample"] for r in rows[0]["rollouts"]),
            [1, 1, 2, 2, 3, 3, 4, 4],
        )

    async def test_total_context_clips_each_independent_trajectory(self):
        requests = []

        async def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            self.assertLessEqual(
                len(body.get("prompt", [101, 102, 103])) + body["max_tokens"], 16
            )
            return httpx.Response(200, stream=Stream(capped(body), delay=0))

        rows, _ = await self.exercise(
            handler,
            changes={
                "max_context_tokens": 16,
                "first_pass_max_tokens": 2,
                "max_rollout_tokens": 32,
            },
        )
        self.assertEqual([b["max_tokens"] for b in requests], [2, 11] * 4)
        self.assertTrue(
            all(r["trajectory_generated_tokens"] <= 13 for r in rows[0]["rollouts"])
        )

    async def test_natural_stop_launches_fresh_without_continuation(self):
        requests = []

        async def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            return httpx.Response(
                200, stream=Stream([chunk("thinking", finish="stop")], delay=0)
            )

        rows, allocation = await self.exercise(
            handler, changes={"max_context_tokens": 65536}
        )
        self.assertEqual(len(requests), 4)
        self.assertTrue(
            all("messages" in b and b["max_tokens"] == 8192 for b in requests)
        )
        self.assertEqual(allocation["questions"]["1"]["fresh"], 4)
        self.assertEqual(rows[0]["status"], "unsolved")

    async def test_default_8k_has_one_remaining_56k_continuation(self):
        requests = []

        async def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            return httpx.Response(200, stream=Stream(capped(body), delay=0))

        rows, allocation = await self.exercise(
            handler,
            changes={"max_context_tokens": 65536, "max_fresh_samples_per_question": 1},
        )
        self.assertEqual([b["max_tokens"] for b in requests], [8192, 57341])
        self.assertEqual(len(requests[1]["prompt"]), 8195)
        self.assertEqual(rows[0]["rollouts"][-1]["trajectory_generated_tokens"], 65533)
        self.assertEqual(allocation["questions"]["1"]["requests"], 2)

    async def test_target_cancels_all_streams_and_recycles_solved_slots(self):
        requests, streams = [], []

        async def handler(request):
            body = json.loads(request.content)
            if request.url.path == "/verify":
                await asyncio.sleep(0.003)
                return httpx.Response(200, json={"verdict": True})
            requests.append(body)
            stream = Stream(
                [chunk("\\boxed{070}" if len(requests) == 1 else "thinking")], hang=True
            )
            streams.append(stream)
            return httpx.Response(200, stream=stream)

        rows, allocation = await self.exercise(handler, count=3)
        self.assertEqual(sum(r["status"] == "solved" for r in rows), 1)
        self.assertTrue(all(s.closed for s in streams))
        self.assertEqual(allocation["peak_active_requests"], 3)
        self.assertTrue(
            all(a["active_requests"] <= 3 for a in allocation["admissions"])
        )

    async def test_freed_slots_wait_for_initial_barrier_then_launch_siblings(self):
        streams, requests, checks = [], [], []

        async def handler(request):
            body = json.loads(request.content)
            if request.url.path == "/verify":
                checks.append(body)
                await asyncio.sleep(0.001)
                return httpx.Response(200, json={"verdict": True})
            q = int(body["messages"][1]["content"])
            requests.append(q)
            text = "\\boxed{070}" if q == 1 or requests.count(q) > 1 else "thinking"
            first = requests.count(q) == 1
            if not first:
                self.assertTrue(all(s.closed for s in streams[:3]))
            stream = Stream(
                [chunk(text, finish="stop" if first and q != 1 else None)],
                hang=not first or q == 1,
                delay=0.025 if first and q != 1 else 0,
            )
            streams.append(stream)
            return httpx.Response(200, stream=stream)

        rows, allocation = await self.exercise(
            handler, count=3, changes={"target_correct": 2}
        )
        self.assertEqual(requests[:3], [1, 2, 3])
        self.assertGreater(len(requests), 3)
        self.assertGreaterEqual(sum(r["status"] == "solved" for r in rows), 2)
        self.assertTrue(all(s.closed for s in streams))
        self.assertTrue(all(s["fresh"] <= 4 for s in allocation["questions"].values()))
        self.assertTrue(
            all(a["phase"] == "coverage" for a in allocation["admissions"][:3])
        )
        release = allocation["barrier_release"]["elapsed_s"]
        self.assertTrue(
            all(
                a["phase"] == "pool" and a["elapsed_s"] >= release
                for a in allocation["admissions"][3:]
            )
        )

    async def test_barrier_waits_for_queued_wrong_verdict_and_ignores_extra_slots(self):
        requests, verdict_done = [], False

        async def handler(request):
            nonlocal verdict_done
            body = json.loads(request.content)
            if request.url.path == "/verify":
                await asyncio.sleep(0.025)
                verdict_done = True
                return httpx.Response(200, json={"verdict": False})
            if len(requests) >= 2:
                self.assertTrue(verdict_done)
            requests.append(body)
            return httpx.Response(
                200, stream=Stream(capped(body, "\\boxed{070}"), delay=0)
            )

        rows, allocation = await self.exercise(
            handler,
            count=2,
            changes={
                "max_concurrent_requests": 4,
                "first_pass_max_tokens": 2,
                "max_rollout_tokens": 6,
                "max_fresh_samples_per_question": 1,
            },
        )
        self.assertEqual(len(requests), 4)
        self.assertEqual(
            [a["phase"] for a in allocation["admissions"]],
            ["coverage"] * 2 + ["pool"] * 2,
        )
        self.assertTrue(
            all(s["coverage_settled"] for s in allocation["questions"].values())
        )

    async def test_missing_exact_ids_and_service_errors_fail_fast(self):
        for fail in ["ids", "inference", "grader"]:

            async def handler(request):
                if request.url.path == "/verify":
                    return httpx.Response(503)
                if fail == "inference":
                    return httpx.Response(500)
                return httpx.Response(
                    200,
                    stream=Stream(
                        [
                            chunk(
                                "\\boxed{070}" if fail == "grader" else "thinking",
                                finish="length",
                            )
                        ]
                    ),
                )

            with self.subTest(fail=fail), self.assertRaises(RuntimeError):
                await self.exercise(handler)

    async def test_prompt_counts_and_input_sized_slot_pool(self):
        args = parse_args([])
        args.max_context_tokens = 65536

        async def handler(request):
            self.assertEqual(request.url.path, "/tokenize")
            return httpx.Response(200, json={"count": 300})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            problems = await prepare_problems(
                [dict(problem_idx=7, problem="a"), dict(problem_idx=9, problem="b")],
                args,
                client,
            )
        self.assertEqual(args.max_concurrent_requests, 2)
        self.assertEqual([p["prompt_tokens"] for p in problems], [300, 300])

    async def test_grader_completion_after_generation_ends_is_not_lost(self):
        streams = []

        async def handler(request):
            if request.url.path == "/verify":
                await asyncio.sleep(0.01)
                return httpx.Response(200, json={"verdict": True})
            stream = Stream([chunk("\\boxed{070}", finish="stop")], delay=0)
            streams.append(stream)
            return httpx.Response(200, stream=stream)

        rows, allocation = await self.exercise(handler)
        self.assertEqual(rows[0]["status"], "solved")
        self.assertEqual(rows[0]["unique_candidates"], 1)
        self.assertLessEqual(allocation["questions"]["1"]["fresh"], 4)
        self.assertTrue(all(s.closed for s in streams))

    async def test_lifecycle_timer_starts_after_tokenization_and_warmup(self):
        from reasoning_speedrun.lib import policy_runtime as runtime
        from reasoning_speedrun.lib.metadata import validate_metadata
        from reasoning_speedrun.extensions.v1_6.cli import metadata, RUNNER_ID
        from tests.helpers import fake_provenance as dataset_provenance

        args = parse_args(["--questions", "1", "--target-correct", "1"])
        events = []

        async def handler(request):
            if request.url.path == "/tokenize":
                events.append("tokenize")
                return httpx.Response(200, json={"count": 3})
            if request.url.path == "/verify":
                return httpx.Response(200, json={"verdict": True})
            events.append("generate")
            return httpx.Response(
                200,
                stream=Stream(
                    capped(json.loads(request.content), "\\boxed{070}"), delay=0
                ),
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        services = SimpleNamespace(close=AsyncMock())

        async def inference(*values):
            args.max_context_tokens = 65536

        async def warmup(*values):
            events.append("warmup")
            return dict(latency_s=0.0, batch_size=1, tokens_per_request=32)

        def local(a, config):
            config["dataset_provenance"] = dataset_provenance(2025)
            return [dict(problem_idx=1, problem="test")]

        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root = Path(tmp)
            with (
                patch.dict(os.environ, {"SPEEDRUN_HOME": str(root)}),
                patch.object(runtime, "Services", return_value=services),
                patch.object(runtime, "ensure_free"),
                patch.object(
                    runtime,
                    "git_state",
                    return_value=dict(git_commit=None, git_dirty=None),
                ),
                patch.object(runtime, "local_dataset", side_effect=local),
                patch.object(
                    runtime,
                    "read_profile",
                    return_value=(
                        root / "profile.yaml",
                        "max-model-len: 65536",
                        {"max-model-len": 65536},
                    ),
                ),
                patch.object(runtime, "prepare_inference", side_effect=inference),
                patch.object(
                    runtime,
                    "prepare_grader",
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
                    runner_module="reasoning_speedrun.extensions.v1_6",
                    metadata_builder=metadata,
                )
            summary = json.loads((output / "summary.json").read_text())
            config = json.loads((output / "config.json").read_text())
            meta = json.loads((output / "metadata.json").read_text())
            self.assertEqual(events[:3], ["tokenize", "warmup", "generate"])
            self.assertEqual(summary["runner_id"], RUNNER_ID)
            self.assertTrue(summary["target_reached"])
            self.assertTrue(summary["trace_storage"]["buffered"])
            self.assertEqual(config["max_concurrent_requests"], 1)
            self.assertEqual(summary["performance"]["fresh_samples"], 1)
            self.assertIsNone(
                meta["controls"]["hyperparameters"]["max_attempts_per_question"]
            )
            self.assertEqual(
                meta["controls"]["hyperparameters"]["max_fresh_samples_per_question"], 4
            )
            validate_metadata(meta, output.name)
            services.close.assert_awaited_once()

    def test_cli_cap_is_fresh_only_and_old_round_knobs_are_rejected(self):
        args = parse_args(["--max-attempts-per-question", "3"])
        self.assertEqual(args.max_fresh_samples_per_question, 3)
        self.assertIsNone(args.max_attempts_per_question)
        self.assertIsNone(args.max_rounds)
        self.assertEqual(args.max_rollout_tokens, 65536)
        self.assertEqual(args.max_tokens, 65536)
        self.assertEqual(args.model_profile, "vllm-v1_6-long64k.yaml")
        self.assertEqual(
            parse_args(["--max-tokens", "32000"]).max_rollout_tokens, 32000
        )
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(["--model-profile", "../invalid.yaml"])
        for argv in [
            ["--max-fresh-samples-per-question", "5"],
            ["--max-rollout-tokens", "65537"],
            ["--max-rounds", "4"],
            ["--no-continuation"],
        ]:
            with (
                self.subTest(argv=argv),
                redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                parse_args(argv)

    def test_manifest_pins_policy_and_flags_drift(self):
        from reasoning_speedrun.lib.common import PACKAGE
        from reasoning_speedrun.integrity import verify_core as verify_base
        from reasoning_speedrun.extensions.v1_6.integrity import verify_core

        base = verify_base(PACKAGE)
        with patch.dict(os.environ, {"SPEEDRUN_STRICT_INTEGRITY": "1"}):
            self.assertEqual(len(verify_core(PACKAGE)), 64)
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                folder = root / "extensions/v1_6"
                folder.mkdir(parents=True)
                manifest = json.loads(
                    (PACKAGE / "extensions/v1_6/manifest.json").read_text()
                )
                (folder / "manifest.json").write_text(json.dumps(manifest))
                with (
                    patch(
                        "reasoning_speedrun.extensions.v1_6.integrity.verify_base",
                        return_value=base,
                    ),
                    self.assertRaisesRegex(RuntimeError, "source drift"),
                ):
                    verify_core(root)


if __name__ == "__main__":
    unittest.main()
