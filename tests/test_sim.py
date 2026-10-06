"""Simulated backend: config, scripted model, serving behavior, and end-to-end runs."""

import asyncio
import io
import json
import os
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import httpx

from qed import cli
from qed.lib.extraction import CandidateDetector
from qed.lib.metadata import validate_metadata
from qed.lib.metrics import parse_engine_metrics
from qed.sim.config import SimConfig, expected, tiers
from qed.sim.engine import Engine
from qed.sim.model import build_plan, plan_rng
from qed.sim.transport import SimulatedTransport
from qed.viewer.store import AttemptStore

EXAMPLES = Path(__file__).resolve().parents[1] / "src/qed/examples"
URL = "http://127.0.0.1:8000"
QUESTION = "What is 2+2?"


def fast(*overrides):
    base = ["decode_tps=1e7", "prefill_tps=1e9", "behavior.difficulty=uniform",
            "behavior.reasoning_median_tokens=300", "behavior.reasoning_min_tokens=60"]
    return SimConfig.from_sources(None, base + list(overrides))


def engine(config=None, key=None, **kw):
    return Engine(config or fast(), "org/model", {QUESTION: 4} if key is None else key,
                  max_model_len=kw.pop("max_model_len", 8192), max_new_tokens=kw.pop("max_new_tokens", None), **kw)


def chat(seed=1, max_tokens=4096, **extra):
    return {"model": "org/model", "messages": [{"role": "system", "content": "Be brief."},
            {"role": "user", "content": QUESTION}], "stream": True, "max_tokens": max_tokens,
            "seed": seed, "return_token_ids": True, **extra}


async def stream(client, path, body):
    """Collect a stream: (text by part, token ids, prompt ids, finish, last usage, chunk count)."""
    text, ids, prompt, finish, usage, chunks = {"reasoning": "", "content": ""}, [], None, None, None, 0
    async with client.stream("POST", URL + path, json=body) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            data = json.loads(line[6:])
            chunks += 1
            prompt = data.get("prompt_token_ids") or prompt
            usage = data.get("usage") or usage
            for choice in data["choices"]:
                ids += choice.get("token_ids", [])
                prompt = choice.get("prompt_token_ids") or prompt
                delta = choice.get("delta") or {"content": choice.get("text", "")}
                text["reasoning"] += delta.get("reasoning_content", "")
                text["content"] += delta.get("content", "")
                finish = choice.get("finish_reason") or finish
    return text, ids, prompt, finish, usage, chunks


def client_for(eng):
    return httpx.AsyncClient(transport=SimulatedTransport(eng, 8000, httpx.AsyncHTTPTransport()))


class ConfigTests(unittest.TestCase):
    def test_defaults_overrides_and_numeric_strings(self):
        config = SimConfig.from_sources(None, ["decode_tps=80", "behavior.p_correct=0.5", "prefill_tps=1e9"])
        self.assertEqual((config.decode_tps, config.behavior.p_correct, config.prefill_tps), (80, 0.5, 1e9))
        self.assertEqual(SimConfig.from_dict(config.to_dict()), config)

    def test_file_then_override_precedence(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sim.yaml"
            path.write_text("decode_tps: 50\nbehavior:\n  p_correct: 0.9\n  answer_at: [0.1, 0.2]\n")
            config = SimConfig.from_sources(path, ["decode_tps=70"])
            self.assertEqual((config.decode_tps, config.behavior.p_correct, config.behavior.answer_at), (70, 0.9, [0.1, 0.2]))
        # The documented all-defaults file must itself be valid and equal the defaults.
        self.assertEqual(SimConfig.from_sources(EXAMPLES / "sim_default.yaml"), SimConfig())

    def test_typos_and_bad_values_are_rejected_with_guidance(self):
        with self.assertRaisesRegex(ValueError, "Unknown simulation keys: decode_tpz"):
            SimConfig.from_sources(None, ["decode_tpz=1"])
        with self.assertRaisesRegex(ValueError, "behavior.p_corect"):
            SimConfig.from_sources(None, ["behavior.p_corect=1"])
        for bad in (["decode_tps=0"], ["max_num_seqs=-1"], ["behavior.p_correct=1.5"],
                    ["behavior.answer_at=[0.9, 0.1]"], ["gpu_memory_utilization=2"], ["nonsense"],
                    ["x.y=1"]):
            with self.assertRaises(ValueError, msg=bad):
                SimConfig.from_sources(None, bad)


class DifficultyTests(unittest.TestCase):
    KEY = {f"Question {i}": i for i in range(1, 31)}

    def eng(self, *overrides, key=None):
        return Engine(SimConfig.from_sources(None, list(overrides)), "org/model", self.KEY if key is None else key, max_model_len=8192)

    def test_mixed_is_the_default_and_uniform_is_one_tier_equal_to_the_flat_knobs(self):
        self.assertEqual(SimConfig().behavior.difficulty, "mixed")
        self.assertEqual([t["name"] for t in tiers(SimConfig().behavior)], ["easy", "medium", "hard"])
        for spec in ("uniform", None):
            config = SimConfig.from_dict({"behavior": {"difficulty": spec, "p_correct": 0.3}})
            (only,) = tiers(config.behavior)
            self.assertEqual((only["weight"], only["behavior"].p_correct), (1.0, 0.3))

    def test_flat_configuration_reproduces_the_pre_tier_plans_exactly(self):
        config = SimConfig.from_sources(None, ["behavior.difficulty=uniform"])
        eng = Engine(config, "m", self.KEY, max_model_len=8192)
        for seed in range(10):
            reference = build_plan(config.behavior, plan_rng(0, seed, "Question 3"), 3)
            plan = eng.plan_for("Question 3", seed)
            self.assertEqual(plan.render(0, plan.total), reference.render(0, reference.total))

    def test_mixed_changes_when_answers_appear_never_whether_they_are_right(self):
        for p in (0.0, 0.1, 0.5, 0.65, 0.99, 1.0):
            mixed = expected(SimConfig.from_sources(None, [f"behavior.p_correct={p}"]).behavior)
            self.assertEqual({tier["p_correct"] for tier in mixed["tiers"]}, {float(p)})
            self.assertAlmostEqual(mixed["mean_p_correct"], p)
        mixed = expected(SimConfig().behavior)
        # Independent correctness: four samples all wrong is (1 - p)^4, whatever the tiers.
        self.assertAlmostEqual(mixed["p_four_samples_all_wrong"], 0.35**4)
        self.assertAlmostEqual(mixed["p_question_solvable_in_four_samples"], 1 - 0.35**4)
        self.assertGreater(30 * mixed["p_question_solvable_in_four_samples"], 18)
        # What the tiers do change: length and the position of the first answer, in order.
        lengths = [t["reasoning_median_tokens"] for t in mixed["tiers"]]
        first = [t["typical_first_answer_tokens"] for t in mixed["tiers"]]
        self.assertTrue(lengths[0] < lengths[1] < lengths[2], lengths)
        self.assertTrue(first[0] < first[1] < first[2], first)
        windows = [t["answer_at"] for t in mixed["tiers"]]
        self.assertTrue(windows[0][0] < windows[1][0] < windows[2][0], windows)

    def test_mixed_is_calibrated_to_the_flat_length_and_answer_window(self):
        for median in (1000, 6000, 20000):
            flat = expected(SimConfig.from_sources(None, ["behavior.difficulty=uniform", f"behavior.reasoning_median_tokens={median}"]).behavior)
            mixed = expected(SimConfig.from_sources(None, [f"behavior.reasoning_median_tokens={median}"]).behavior)
            self.assertAlmostEqual(mixed["mean_reasoning_tokens"] / flat["mean_reasoning_tokens"], 1, delta=0.01)
        # The medium tier is the flat answer window; easy shifts earlier and hard later.
        config = SimConfig.from_sources(None, ["behavior.answer_at=[0.5, 0.6]"])
        easy, medium, hard = (t["answer_at"] for t in expected(config.behavior)["tiers"])
        self.assertEqual(medium, [0.5, 0.6])
        self.assertTrue(easy[1] < medium[0] + 0.1 and hard[0] > medium[0])
        # Shifted windows stay inside [0, 1] and ordered at the edges.
        for window in ("[0.0, 0.05]", "[0.95, 1.0]"):
            for tier in expected(SimConfig.from_sources(None, [f"behavior.answer_at={window}"]).behavior)["tiers"]:
                low, high = tier["answer_at"]
                self.assertTrue(0 <= low <= high <= 1, (window, tier))

    def test_flat_knobs_do_not_override_an_explicit_tier_list(self):
        config = SimConfig.from_sources(None, ["behavior.p_correct=0.2",
                                               "behavior.difficulty=[{weight: 1, p_correct: 0.9}, {weight: 1}]"])
        explicit, inherited = tiers(config.behavior)
        self.assertEqual((explicit["behavior"].p_correct, inherited["behavior"].p_correct), (0.9, 0.2))

    def test_tier_overrides_inherit_everything_else_and_weights_normalize(self):
        config = SimConfig.from_sources(None, ["behavior.p_wrong_first=0.4", "behavior.reasoning_sigma=0.3",
                                               "behavior.difficulty=[{weight: 1, p_correct: 0.2}, {weight: 3, name: easy, p_correct: 0.9, answer_at: [0.1, 0.2]}]"])
        first, second = tiers(config.behavior)
        self.assertEqual((first["weight"], second["weight"]), (0.25, 0.75))
        self.assertEqual((first["name"], second["name"]), ("tier1", "easy"))
        self.assertEqual((first["behavior"].p_correct, first["behavior"].p_wrong_first, first["behavior"].reasoning_sigma), (0.2, 0.4, 0.3))
        self.assertEqual((second["behavior"].answer_at, second["behavior"].p_wrong_first), ([0.1, 0.2], 0.4))

    def test_bad_tiers_are_rejected_with_guidance(self):
        for bad in ("behavior.difficulty=nope", "behavior.difficulty=[]", "behavior.difficulty=[{p_correct: 0.5}]",
                    "behavior.difficulty=[{weight: 1, bogus: 2}]", "behavior.difficulty=[{weight: 1, p_correct: 3}]",
                    "behavior.difficulty=[{weight: 0}]", "behavior.difficulty=[1]",
                    "behavior.difficulty=[{weight: 1, answer_at: [0.9, 0.1]}]"):
            with self.assertRaises(ValueError, msg=bad):
                SimConfig.from_sources(None, [bad])
        with self.assertRaisesRegex(ValueError, "mixed"):
            SimConfig.from_sources(None, ["behavior.difficulty=nope"])

    def test_assignment_matches_the_weights_exactly_and_is_stable(self):
        eng = self.eng("behavior.difficulty=mixed")
        counts = [sum(1 for t in eng.tier_of.values() if t == i) for i in range(3)]
        self.assertEqual(counts, [14, 10, 6])  # 7:5:3 of 30 questions
        # The same question has the same tier whichever subset of questions a run selects.
        again = self.eng("behavior.difficulty=mixed", key={k: v for k, v in list(self.KEY.items())[:30]})
        self.assertEqual(eng.tier_of, again.tier_of)
        # ...and across request seeds: difficulty belongs to the question, not the sample.
        self.assertEqual({eng.tier_index("Question 7")}, {eng.tier_index("Question 7") for _ in range(3)})
        other = self.eng("behavior.difficulty=mixed", "behavior.seed=5")
        self.assertNotEqual(eng.tier_of, other.tier_of)

    def test_largest_remainder_handles_awkward_counts(self):
        eng = self.eng("behavior.difficulty=[{weight: 1}, {weight: 1}, {weight: 1}]", key={f"q{i}": i for i in range(10)})
        self.assertEqual(sorted(sum(1 for t in eng.tier_of.values() if t == i) for i in range(3)), [3, 3, 4])

    def test_tiers_differ_in_length_and_first_answer_position_but_not_accuracy(self):
        eng = self.eng("behavior.reasoning_min_tokens=50")
        stats = {0: [], 1: [], 2: []}
        for text in self.KEY:
            plans = [eng.plan_for(text, seed) for seed in range(60)]
            stats[eng.tier_index(text)].append((
                sum(p.reasoning for p in plans) / 60,
                sum(p.final_answer == p.gold for p in plans) / 60,
                sum(min(p.inserts) for p in plans) / 60,  # token position of the first planted answer
            ))
        mean = lambda i, col: sum(row[col] for row in stats[i]) / len(stats[i])
        for column in (0, 2):
            self.assertLess(mean(0, column), mean(1, column))
            self.assertLess(mean(1, column), mean(2, column))
        for tier in range(3):
            self.assertAlmostEqual(mean(tier, 1), 0.65, delta=0.06)

    def test_prompts_outside_the_key_still_get_a_tier(self):
        eng = self.eng("behavior.difficulty=mixed")
        self.assertIn(eng.tier_index("Compute 1 + 1."), {0, 1, 2})
        plan = eng.plan_for("Compute 1 + 1.", 1, min_tokens=32)
        self.assertGreaterEqual(plan.total, 32)

    def test_report_lists_questions_by_tier(self):
        from qed.sim import Key

        key = Key({f"Question {i}": i for i in range(1, 11)})
        key.indices = {f"Question {i}": i for i in range(1, 11)}
        eng = Engine(SimConfig.from_sources(None, ["behavior.difficulty=mixed"]), "m", key, max_model_len=8192)
        for text in key:
            eng.plan_for(text, 1)
        report = eng.report()["difficulty"]
        self.assertEqual([t["name"] for t in report["tiers"]], ["easy", "medium", "hard"])
        listed = [i for v in report["questions_by_tier"].values() for i in v]
        self.assertEqual(sorted(listed), list(range(1, 11)))
        self.assertEqual([len(v) for v in report["questions_by_tier"].values()], [5, 3, 2])


class DescribeTests(unittest.TestCase):
    def test_sim_config_subcommand_prints_resolved_values_and_implications(self):
        from qed.sim import describe

        out = io.StringIO()
        with redirect_stdout(out):
            describe.main(["--sim", "behavior.difficulty=mixed", "--sim", "decode_tps=60"])
        text = out.getvalue()
        self.assertIn("decode_tps: 60", text)
        self.assertIn("name: hard", text)
        self.assertIn("p_four_samples_all_wrong", text)
        self.assertIn("p_question_solvable_in_four_samples", text)
        self.assertNotIn("&id", text)
        self.assertNotIn("0.92026", text)  # tier values are rounded for reading
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            describe.main(["--sim", "decode_tpz=1"])

    def test_routed_from_the_main_cli(self):
        with patch("qed.sim.describe.main") as describe:
            cli.main(["sim-config", "--sim", "decode_tps=1"])
            describe.assert_called_once_with(["--sim", "decode_tps=1"])


class ModelTests(unittest.TestCase):
    def render(self, plan):
        text = {"reasoning": "", "content": ""}
        for part, piece, _ in plan.render(0, plan.total):
            text[part] += piece
        return text

    def detect(self, text):
        d, found = CandidateDetector(), []
        for part in ("reasoning", "content"):
            found += d.feed(part, text[part], eof=True)
        return found

    def test_plans_are_deterministic_and_seed_sensitive(self):
        b = fast().behavior
        a1 = build_plan(b, plan_rng(0, 7, "q"), 70)
        a2 = build_plan(b, plan_rng(0, 7, "q"), 70)
        other = build_plan(b, plan_rng(0, 8, "q"), 70)
        self.assertEqual(a1.render(0, a1.total), a2.render(0, a2.total))
        self.assertNotEqual((a1.total, a1.final_answer), (other.total, other.final_answer))

    def test_planted_answers_use_forms_the_real_extractor_recognises(self):
        b = SimConfig.from_sources(None, ["behavior.p_correct=1", "behavior.p_wrong_first=0"]).behavior
        kinds = set()
        for seed in range(30):
            plan = build_plan(b, plan_rng(0, seed, "q"), 70)
            found = self.detect(self.render(plan))
            answers = [f["answer"] for f in found]
            self.assertEqual(set(answers), {70}, (seed, found))
            self.assertEqual(found[0]["part"], "reasoning")
            self.assertEqual(found[-1]["part"], "content")
            self.assertEqual(found[-1]["kind"], "boxed")
            kinds.add(found[0]["kind"])
        self.assertEqual(kinds, {"boxed", "answer_line", "literal_prose"})

    def test_accuracy_and_wrong_first_controls(self):
        wrong = SimConfig.from_sources(None, ["behavior.p_correct=0"]).behavior
        self.assertTrue(all(build_plan(wrong, plan_rng(0, s, "q"), 70).final_answer != 70 for s in range(20)))
        first = SimConfig.from_sources(None, ["behavior.p_correct=1", "behavior.p_wrong_first=1"]).behavior
        plan = build_plan(first, plan_rng(0, 1, "q"), 70)
        answers = [f["answer"] for f in self.detect(self.render(plan))]
        self.assertEqual(answers[0] != 70, True)
        self.assertEqual(set(answers[1:]), {70})
        # No answer key: the model can still talk, but is never "correct".
        self.assertNotIn(None, [build_plan(first, plan_rng(0, 1, "q"), None).final_answer])

    def test_minimum_length_and_lazy_rendering(self):
        plan = build_plan(fast().behavior, plan_rng(0, 1, "q"), 70, min_tokens=5000)
        self.assertGreaterEqual(plan.total, 5000)
        self.assertEqual(plan.render(10, 20), plan.render(0, 20)[10:])
        self.assertEqual(plan.render(plan.total - 3, plan.total + 50), plan.render(0, plan.total)[-3:])


class ServingTests(unittest.IsolatedAsyncioTestCase):
    async def test_chat_stream_shape_usage_and_token_ids(self):
        eng = engine()
        async with client_for(eng) as client:
            text, ids, prompt, finish, usage, chunks = await stream(client, "/v1/chat/completions", chat())
        plan = eng.plan_for(QUESTION, 1)
        self.assertEqual(finish, "stop")
        self.assertEqual(len(ids), plan.total)
        self.assertEqual(usage["completion_tokens"], plan.total)
        self.assertEqual(usage["prompt_tokens"], len(prompt))
        self.assertIn("cached_tokens", usage["prompt_tokens_details"])
        self.assertTrue(text["reasoning"] and text["content"].rstrip().endswith("}"))
        self.assertGreater(chunks, plan.total // eng.config.stream_interval_tokens)

    async def test_cap_gives_length_finish_and_exact_continuation_resumes_the_trajectory(self):
        eng = engine(fast("behavior.reasoning_median_tokens=900", "behavior.reasoning_min_tokens=900"))
        async with client_for(eng) as client:
            whole = (await stream(client, "/v1/chat/completions", chat(seed=5)))[1]
            first = await stream(client, "/v1/chat/completions", chat(seed=5, max_tokens=300))
            self.assertEqual((first[3], len(first[1])), ("length", 300))
            self.assertEqual(first[1], whole[:300])
            body = {"model": "org/model", "prompt": first[2] + first[1], "stream": True,
                    "max_tokens": 4000, "return_token_ids": True}
            rest = await stream(client, "/v1/completions", body)
            # Exact IDs in, the rest of the same trajectory out, all as plain text.
            self.assertEqual(first[1] + rest[1], whole)
            self.assertEqual(rest[3], "stop")
            self.assertEqual(rest[0]["reasoning"], "")
            # The continued prompt was just produced, so most of it is a prefix-cache hit.
            self.assertGreaterEqual(rest[4]["prompt_tokens_details"]["cached_tokens"], len(body["prompt"]) - 16)

    async def test_unknown_prefix_and_bad_requests_fail_loudly(self):
        async with client_for(engine()) as client:
            bad = await client.post(URL + "/v1/completions", json={"model": "org/model", "prompt": [1, 2, 3]})
            self.assertEqual(bad.status_code, 400)
            self.assertIn("Unknown token prefix", bad.text)
            self.assertEqual((await client.post(URL + "/v1/completions", json={"model": "org/model", "prompt": "text"})).status_code, 400)
            self.assertEqual((await client.post(URL + "/v1/chat/completions", json={**chat(), "model": "nope"})).status_code, 404)
            tiny = await client.post(URL + "/v1/chat/completions", json=chat())
            self.assertEqual(tiny.status_code, 200)
        small = engine(max_model_len=10)
        async with client_for(small) as client:
            self.assertEqual((await client.post(URL + "/v1/chat/completions", json=chat())).status_code, 400)

    async def test_non_stream_warmup_honors_min_and_max_tokens(self):
        body = {"model": "org/model", "messages": [{"role": "user", "content": "Compute 1 + 1."}],
                "max_tokens": 32, "min_tokens": 32, "stream": False}
        async with client_for(engine()) as client:
            data = (await client.post(URL + "/v1/chat/completions", json=body)).json()
        self.assertEqual(data["usage"]["completion_tokens"], 32)
        self.assertEqual(data["choices"][0]["finish_reason"], "length")

    async def test_generation_ceiling_and_models_endpoint(self):
        eng = engine(max_new_tokens=50)
        async with client_for(eng) as client:
            self.assertEqual((await stream(client, "/v1/chat/completions", chat(max_tokens=4000)))[3], "length")
            models = (await client.get(URL + "/v1/models")).json()
        self.assertEqual(models["data"][0]["max_model_len"], 8192)
        self.assertEqual(models["data"][0]["id"], "org/model")

    async def test_decode_rate_and_prefill_ttft_follow_the_config(self):
        eng = engine(fast("decode_tps=4000", "prefill_tps=500", "behavior.reasoning_median_tokens=800",
                          "behavior.reasoning_min_tokens=500"))
        async with client_for(eng) as client:
            begun = time.perf_counter()
            text, ids, prompt, *_ = await stream(client, "/v1/chat/completions", chat(max_tokens=800))
            elapsed = time.perf_counter() - begun
        prefill, decode = len(prompt) / 500, len(ids) / 4000
        self.assertGreater(len(ids), 200)
        self.assertGreater(elapsed, 0.9 * (prefill + decode))
        self.assertLess(elapsed, prefill + decode + 0.25)
        self.assertGreater(eng.counters["ttft_s"], prefill)

    async def test_max_num_seqs_queues_fifo_and_reports_waiting(self):
        eng = engine(fast("decode_tps=1000", "max_num_seqs=1"))
        spans = []

        async def one(client, seed):
            begun = time.perf_counter()
            await stream(client, "/v1/chat/completions", chat(seed=seed, max_tokens=200))
            spans.append((seed, begun, time.perf_counter()))

        async with client_for(eng) as client:
            await asyncio.gather(one(client, 1), one(client, 2), one(client, 3))
        order = [s for s, _, _ in sorted(spans, key=lambda s: s[2])]
        self.assertEqual(order, [1, 2, 3])
        self.assertEqual(eng.counters["peak_running"], 1)
        self.assertEqual(eng.counters["peak_waiting"], 2)
        self.assertGreater(eng.counters["queue_s"], 0.3)  # ~0.2s + ~0.4s of waiting between the three

    async def test_uncontended_requests_never_count_as_waiting(self):
        eng = engine()
        async with client_for(eng) as client:
            await asyncio.gather(*(stream(client, "/v1/chat/completions", chat(seed=s, max_tokens=50)) for s in range(5)))
        self.assertEqual((eng.counters["peak_waiting"], eng.counters["peak_running"]), (0, 5))

    async def test_cancellation_frees_the_slot_and_the_partial_prefix_can_continue(self):
        eng = engine(fast("decode_tps=2000"))
        async with client_for(eng) as client:
            ids, prompt = [], None
            async with client.stream("POST", URL + "/v1/chat/completions", json=chat(seed=9)) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: ") and line != "data: [DONE]":
                        data = json.loads(line[6:])
                        prompt = data.get("prompt_token_ids") or prompt
                        for choice in data["choices"]:
                            ids += choice.get("token_ids", [])
                        if len(ids) >= 40:
                            break
            await asyncio.sleep(0.05)
            self.assertEqual((eng.running, eng.live_tokens), (0, 0))
            self.assertEqual(eng.counters["cancelled"], 1)
            rest = await stream(client, "/v1/completions", {"model": "org/model", "prompt": prompt + ids,
                                                          "stream": True, "max_tokens": 4000, "return_token_ids": True})
            whole = (await stream(client, "/v1/chat/completions", chat(seed=9)))[1]
            self.assertEqual(ids + rest[1], whole)

    async def test_prefix_cache_shares_the_system_prompt_and_can_be_reset_or_disabled(self):
        key = {QUESTION: 4, "Second question?": 5}
        eng = engine(key=key)
        long_system = "You are a careful mathematician. " * 20
        async with client_for(eng) as client:
            def request(q):
                return {**chat(max_tokens=20), "messages": [{"role": "system", "content": long_system},
                                                            {"role": "user", "content": q}]}
            first = await stream(client, "/v1/chat/completions", request(QUESTION))
            second = await stream(client, "/v1/chat/completions", request("Second question?"))
            self.assertEqual(first[4]["prompt_tokens_details"]["cached_tokens"], 0)
            self.assertGreater(second[4]["prompt_tokens_details"]["cached_tokens"], 100)
            self.assertTrue((await client.post(URL + "/reset_prefix_cache")).json()["success"])
            third = await stream(client, "/v1/chat/completions", request("Second question?"))
            self.assertEqual(third[4]["prompt_tokens_details"]["cached_tokens"], 0)
        off = engine(fast("prefix_cache=false"), key=key)
        async with client_for(off) as client:
            await stream(client, "/v1/chat/completions", request(QUESTION))
            again = await stream(client, "/v1/chat/completions", request(QUESTION))
        self.assertEqual(again[4]["prompt_tokens_details"]["cached_tokens"], 0)

    async def test_reset_refuses_while_a_request_runs(self):
        eng = engine(fast("decode_tps=500"))
        async with client_for(eng) as client:
            task = asyncio.create_task(stream(client, "/v1/chat/completions", chat(max_tokens=200)))
            await asyncio.sleep(0.1)
            self.assertFalse((await client.post(URL + "/reset_prefix_cache")).json()["success"])
            await task
            self.assertTrue((await client.post(URL + "/reset_prefix_cache")).json()["success"])

    async def test_metrics_text_parses_with_the_runners_own_parser(self):
        eng = engine()
        async with client_for(eng) as client:
            await stream(client, "/v1/chat/completions", chat(max_tokens=100))
            rows = parse_engine_metrics((await client.get(URL + "/metrics")).text)
        values = {r["name"].removeprefix("vllm:"): r["value"] for r in rows}
        self.assertEqual(values["generation_tokens_total"], 100)
        self.assertEqual(values["num_requests_running"], 0)
        self.assertGreater(values["prompt_tokens_total"], 0)
        self.assertIn("kv_cache_usage_perc", values)

    async def test_non_simulated_hosts_fall_through_to_the_real_transport(self):
        calls = []

        class Fallback(httpx.AsyncBaseTransport):
            async def handle_async_request(self, request):
                calls.append(str(request.url))
                return httpx.Response(200, json={"grader": True})

        eng = engine()
        client = httpx.AsyncClient(transport=SimulatedTransport(eng, 8000, Fallback()))
        async with client:
            self.assertEqual((await client.get("http://127.0.0.1:8077/health")).json(), {"grader": True})
            await client.get(URL + "/v1/models")
        self.assertEqual(calls, ["http://127.0.0.1:8077/health"])

    def test_gpu_sampler_reports_preallocated_vram_and_activity(self):
        from qed.sim.gpu import SimulatedGPUSampler

        eng = engine(gpu_memory_utilization=0.5)
        with tempfile.TemporaryDirectory() as tmp:
            sampler = SimulatedGPUSampler(Path(tmp) / "gpu.jsonl", eng, interval=0.01)
            begun = time.perf_counter()
            asyncio.run(sampler.start())
            eng.running = 3
            time.sleep(0.06)
            sampler.stop()
            window = sampler.window(begun, time.perf_counter())
            rows = [json.loads(line) for line in (Path(tmp) / "gpu.jsonl").read_text().splitlines()]
        self.assertAlmostEqual(window["observed_peak_vram_mib"], 81920 * 0.5)
        self.assertEqual({r["vram_total_mib"] for r in rows}, {81920.0})
        self.assertIn(100, {r["gpu_util_pct"] for r in rows})
        self.assertIsNone(sampler.error)


class CliTests(unittest.TestCase):
    def test_flags_validate_and_record_the_configuration(self):
        args = cli.parse_args(["--simulate", "--sim", "decode_tps=50", "--sim", "behavior.p_correct=0.4"])
        self.assertTrue(args.simulate)
        self.assertEqual((args.simulation["decode_tps"], args.simulation["behavior"]["p_correct"]), (50, 0.4))
        json.dumps(vars(args))  # everything recorded in config.json must serialize
        self.assertIsNone(cli.parse_args([]).simulation)
        for argv in (["--sim", "decode_tps=1"], ["--sim-config", "x.yaml"], ["--simulate", "--reuse-server"],
                     ["--simulate", "--sim", "decode_tpz=1"], ["--simulate", "--sim-config", "/no/such.yaml"],
                     ["--simulate", "--sim", "decode_tps=0"]):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit, msg=argv):
                cli.parse_args(argv)

    def test_every_policy_accepts_the_simulate_flag(self):
        from qed.extensions.naive.cli import parse_args as naive
        from qed.extensions.v1_6.cli import parse_args as v16

        for parse in (naive, v16):
            self.assertTrue(parse(["--simulate", "--sim", "decode_tps=9"]).simulation["decode_tps"] == 9)


class EndToEndTests(unittest.TestCase):
    """The real runner, scheduler, grader process and traces against the simulated backend."""

    def run_cli(self, *extra, expect_time=True):
        self.stdout = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(self.stdout), patch.dict(os.environ, {"QED_HOME": tmp}), \
                patch("qed.lib.setup.assert_gpu_idle", side_effect=AssertionError("must not touch a GPU")):
            cli.main([
                "--simulate", "--grader-config", str(EXAMPLES / "synthetic_grader.yaml"),
                "--system-prompt-file", str(EXAMPLES / "integer_prompt.txt"),
                "--questions", "1", "2", "3", "4", "--target-correct", "2", "--grader-cost", "0.05",
                "--sim", "decode_tps=2e5", "--sim", "prefill_tps=1e8",
                "--sim", "behavior.reasoning_median_tokens=1500", "--sim", "behavior.p_correct=0.9", *extra,
            ])
            (folder,) = sorted(Path(tmp, "attempts").iterdir())
            read = lambda name: json.loads((folder / name).read_text())
            result = {name: read(name) for name in ("summary.json", "config.json", "metadata.json", "simulation.json")}
            validate_metadata(result["metadata.json"], folder.name)
            result["listed"] = AttemptStore(Path(tmp, "attempts")).list()
            result["files"] = sorted(p.name for p in folder.iterdir())
            return result

    def check(self, result, runner_id):
        summary, config, meta, sim = (result[k] for k in ("summary.json", "config.json", "metadata.json", "simulation.json"))
        self.assertEqual((summary["status"], summary["target_reached"], summary["runner_id"]), ("completed", True, runner_id))
        self.assertGreater(summary["time_to_target_s"], 0)
        self.assertTrue(config["simulated"])
        self.assertEqual(config["simulation"]["resolved"]["max_model_len"], 65536)
        self.assertNotIn("vllm_command", config)
        self.assertEqual(meta["gpu"]["device"], "simulated")
        self.assertTrue(meta["label"].endswith("· simulated"), meta["label"])  # shown on the results page
        self.assertEqual(meta["simulation"]["decode_tps"], 2e5)
        self.assertIn("difficulty", sim)
        self.assertGreaterEqual(sim["requests"], 4)
        self.assertGreater(sim["generation_tokens"], 0)
        self.assertNotIn("vllm.log", result["files"])
        self.assertTrue(result["listed"]["attempts"][0]["simulated"])
        # The last thing a person sees is the verdict, not the JSON dump above it.
        last = self.stdout.getvalue().rstrip().splitlines()
        self.assertIn(f"reached {summary['target_correct']} correct in {summary['time_to_target_s']:.1f} s", "\n".join(last[-3:]))
        self.assertTrue(last[-1].strip().endswith("browse it with `qed view`"), last[-1])
        self.assertIn("simulated backend", "\n".join(last[-2:]))

    def test_canonical_v1_with_continuations_against_the_simulator(self):
        # Solving all four needs answers that appear after a tiny first request: exact-ID continuations.
        result = self.run_cli("--target-correct", "4", "--first-pass-max-tokens", "400", "--sim", "behavior.p_correct=1")
        self.check(result, "runner_core_v1")
        self.assertGreater(result["summary.json"]["performance"]["generation_requests"], 4)

    def test_naive_policy_against_the_simulator(self):
        result = self.run_cli("--version", "naive", "--samples-per-question", "2")
        self.check(result, "runner_core_naive")

    def test_v1_6_policy_against_the_simulator(self):
        self.check(self.run_cli("--version", "v1.6"), "runner_core_v1_6")

    def test_profile_mode_records_simulated_gpu_and_engine_samples(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), patch.dict(os.environ, {"QED_HOME": tmp}):
            cli.main([
                "--simulate", "--profile", "--grader-config", str(EXAMPLES / "synthetic_grader.yaml"),
                "--system-prompt-file", str(EXAMPLES / "integer_prompt.txt"),
                "--questions", "1", "2", "--target-correct", "1", "--grader-cost", "0.05",
                "--engine-metrics-interval", "0.05", "--gpu-interval", "0.02",
                "--sim", "decode_tps=2000", "--sim", "prefill_tps=1e8",
                "--sim", "behavior.reasoning_median_tokens=800", "--sim", "behavior.p_correct=1",
            ])
            (folder,) = sorted(Path(tmp, "attempts").iterdir())
            gpu = [json.loads(l) for l in (folder / "gpu.jsonl").read_text().splitlines()]
            engine_rows = [json.loads(l) for l in (folder / "inference_metrics.jsonl").read_text().splitlines()]
            summary = json.loads((folder / "summary.json").read_text())
        self.assertTrue(gpu and all(r["vram_total_mib"] == 81920 for r in gpu))
        self.assertTrue(any(r["gpu_util_pct"] == 100 for r in gpu))
        self.assertTrue(engine_rows)
        names = {m["name"] for row in engine_rows for m in row["metrics"]}
        self.assertIn("vllm:num_requests_running", names)
        self.assertIsNotNone(summary["performance"]["official_gpu"]["observed_peak_vram_mib"])

    def test_mixed_difficulty_run_records_which_questions_were_hard(self):
        result = self.run_cli("--sim", "behavior.difficulty=mixed")
        self.check(result, "runner_core_v1")
        tiers_seen = result["simulation.json"]["difficulty"]["questions_by_tier"]
        self.assertEqual(set(tiers_seen), {"easy", "medium", "hard"})
        self.assertEqual(sorted(i for v in tiers_seen.values() for i in v), [1, 2, 3, 4])
        resolved = result["config.json"]["simulation"]["resolved"]["difficulty_tiers"]
        self.assertEqual([t["name"] for t in resolved], ["easy", "medium", "hard"])

    def test_relative_dataset_source_resolves_against_the_config_file(self):
        with tempfile.TemporaryDirectory() as tmp, tempfile.TemporaryDirectory() as home, redirect_stdout(io.StringIO()), \
                patch.dict(os.environ, {"QED_HOME": home}):
            shutil_copy = __import__("shutil").copy
            shutil_copy(EXAMPLES / "synthetic_30.jsonl", Path(tmp) / "my_questions.jsonl")
            (Path(tmp) / "grader.yaml").write_text(
                "dataset:\n  source: my_questions.jsonl\n  format: jsonl\n  idx_field: problem_idx\n"
                "  problem_field: problem\n  gold_field: answer\n  id: mine\ncost_c: 3.0\nhost: 127.0.0.1\nport: 8077\n")
            cli.main(["--simulate", "--grader-config", str(Path(tmp) / "grader.yaml"),
                      "--system-prompt-file", str(EXAMPLES / "integer_prompt.txt"), "--questions", "1", "2",
                      "--target-correct", "1", "--grader-cost", "0.05", "--sim", "decode_tps=2e5",
                      "--sim", "behavior.reasoning_median_tokens=800", "--sim", "behavior.p_correct=1"])
            (folder,) = sorted(Path(home, "attempts").iterdir())
            summary = json.loads((folder / "summary.json").read_text())
            copied = (folder / "grader_config.yaml").read_text()
        self.assertTrue(summary["target_reached"])
        self.assertIn(str(Path(tmp).resolve() / "my_questions.jsonl"), copied)  # recorded as an absolute path

    def test_bundled_configs_with_grader_dir_relative_sources_still_resolve(self):
        from qed.lib.datasets import grader_dataset

        for name in ("integer_grader.yaml", "synthetic_grader.yaml"):
            source = Path(grader_dataset(EXAMPLES / name)["dataset"]["source"])
            self.assertTrue(source.is_absolute() and source.is_file(), (name, source))

    def test_missing_answer_key_is_a_clear_error(self):
        from qed.sim import answer_key

        args = cli.parse_args(["--simulate", "--reuse-grader"])
        with self.assertRaisesRegex(ValueError, "behavior.answers"):
            answer_key(args, SimConfig.from_dict(args.simulation))
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {"QED_DATA": tmp}):
            args = cli.parse_args(["--simulate"])
            with self.assertRaisesRegex(FileNotFoundError, "fetch-data"):
                answer_key(args, SimConfig.from_dict(args.simulation))

    def test_answer_key_from_an_explicit_file(self):
        from qed.sim import answer_key

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "key.jsonl"
            path.write_text('{"problem": "P?", "answer": 7}\n{"problem": "Q?", "answer": "12"}\n')
            args = cli.parse_args(["--simulate", "--reuse-grader", "--sim", f"behavior.answers={path}"])
            self.assertEqual(answer_key(args, SimConfig.from_dict(args.simulation)), {"P?": 7, "Q?": 12})


if __name__ == "__main__":
    unittest.main()
