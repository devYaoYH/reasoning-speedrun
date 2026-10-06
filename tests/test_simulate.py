"""Deterministic replay: hand-computed scenarios and reproduction of the bundled example."""

from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout

from reasoning_speedrun import simulate

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/attempts/20261004T220514.018752Z"
ORIGIN = datetime(2026, 1, 1, tzinfo=timezone.utc)


def iso(seconds):
    return (ORIGIN + timedelta(seconds=seconds)).isoformat()


def build(root, trajectories, *, target=2, cost=3.0):
    """trajectories: {question: [dict(rollout, start, end, finish, candidates=[(answer, t, verdict)])]}"""
    (root / "config.json").write_text(
        json.dumps({"official_started_at_utc": iso(0), "target_correct": target, "grader_cost": cost})
    )
    for question, rows in trajectories.items():
        folder = root / "trace" / f"{question:02d}"
        folder.mkdir(parents=True)
        rollouts, events = [], []
        for row in rows:
            rollouts.append(
                {
                    "rollout": row["rollout"],
                    "started_at_utc": iso(row["start"]),
                    "generation_finished_at_utc": iso(row["end"]),
                    "generation_censored": row["finish"] != "stop",
                    "finish_reason": row["finish"],
                    "continuation_of_rollout": row.get("parent"),
                    "usage": {"completion_tokens": 100},
                }
            )
            for answer, t, verdict in row["candidates"]:
                event = {"candidate": answer, "rollout": row["rollout"], "kind": "boxed", "observed_at_utc": iso(t)}
                if verdict is not None:
                    event["result"] = {"verdict": verdict}
                events.append(event)
        (folder / "question.json").write_text(json.dumps({"problem_idx": question, "rollouts": rollouts}))
        (folder / "verification.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (root / "solved.jsonl").write_text("")
    return simulate.load_trajectories(root)


def standard(root):
    return build(
        root,
        {
            1: [
                dict(rollout=1, start=0, end=10, finish="stop", candidates=[("70", 4, True)]),
                dict(rollout=2, start=0, end=20, finish="stop", candidates=[]),
            ],
            2: [dict(rollout=1, start=0, end=12, finish="stop", candidates=[("5", 2, False), ("9", 6, True)])],
        },
    )


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = standard(Path(self.tmp.name))

    def test_recorded_policy_early_extraction_fifo_verifier(self):
        # 5 wrong 2-5, 70 right 5-8 (Q1), 9 right 8-11 (Q2).
        result = simulate.simulate(self.data)
        self.assertEqual(result["time_to_target_s"], 11)
        self.assertEqual((result["checks"], result["wrong_checks"]), (3, 1))
        self.assertEqual([m["problem_idx"] for m in result["milestones"]], [1, 2])
        # Solving Q1 at 8 cancels its second trajectory at 8, not 20.
        self.assertEqual(result["trajectories_started"], 3)
        self.assertAlmostEqual(result["stream_seconds"], 8 + 8 + 11)

    def test_verifier_cost_scales_queueing(self):
        # Checks occupy 2-3, 3-4 (wait for 70 at 4: 4-5), then 9 at 6: 6-7 -> done at 7.
        self.assertEqual(simulate.simulate(self.data, cost_s=1)["time_to_target_s"], 7)

    def test_final_only_waits_for_trajectory_end(self):
        # Last answers arrive at 10 (70) and 12 (9): checks 10-13, 13-16.
        result = simulate.simulate(self.data, extraction="final")
        self.assertEqual(result["time_to_target_s"], 16)
        self.assertEqual(result["checks"], 2)
        self.assertEqual(result["wrong_checks"], 0)

    def test_bounded_slots_serialize_admission(self):
        # One slot, FIFO by start: Q1/s1 solves at 7 (candidate 4 + 3s) and frees the slot;
        # Q1/s2 is skipped as solved; Q2 starts at 7 -> '5' arrives 9, '9' arrives 13 -> solved at 16.
        result = simulate.simulate(self.data, slots=1)
        self.assertEqual(result["time_to_target_s"], 16)
        self.assertEqual(result["peak_concurrent_streams"], 1)
        self.assertEqual(result["trajectories_started"], 2)

    def test_no_cancel_keeps_streams_running(self):
        kept = simulate.simulate(self.data, cancel_on_solve=False)
        cancelled = simulate.simulate(self.data)
        self.assertEqual(kept["time_to_target_s"], cancelled["time_to_target_s"])
        self.assertGreater(kept["stream_seconds"], cancelled["stream_seconds"])

    def test_unmet_target_and_validation(self):
        result = simulate.simulate(self.data, target=3)
        self.assertIsNone(result["time_to_target_s"])
        self.assertEqual(result["solved"], 2)
        for bad in (dict(slots=0), dict(cost_s=0), dict(extraction="x"), dict(censored="x")):
            with self.assertRaises(ValueError):
                simulate.simulate(self.data, **bad)

    def test_deterministic(self):
        a = simulate.simulate(self.data, slots=2, extraction="final")
        b = simulate.simulate(self.data, slots=2, extraction="final")
        self.assertEqual(a, b)


class CensoringAndUnknownTests(unittest.TestCase):
    def test_censored_final_answers_are_bracketed(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = build(
                Path(tmp),
                {
                    1: [dict(rollout=1, start=0, end=9, finish=None, candidates=[("70", 4, True)])],
                    2: [dict(rollout=1, start=0, end=5, finish="stop", candidates=[("9", 3, True)])],
                },
            )
            dropped = simulate.simulate(data, extraction="final", censored="drop")
            bound = simulate.simulate(data, extraction="final", censored="lower_bound")
            self.assertIsNone(dropped["time_to_target_s"])  # Q1's final answer never observed
            self.assertEqual(dropped["solved"], 1)
            # Lower bound: 9 at 5 -> 5-8; 70 at cancel time 9 -> 9-12.
            self.assertEqual(bound["time_to_target_s"], 12)
            early = simulate.simulate(data, extraction="early")
            self.assertLessEqual(early["time_to_target_s"], bound["time_to_target_s"])

    def test_candidate_without_verdict_is_skipped_not_guessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = build(
                Path(tmp),
                {1: [dict(rollout=1, start=0, end=9, finish="stop", candidates=[("70", 1, None), ("71", 2, True)])]},
                target=1,
            )
            result = simulate.simulate(data)
            self.assertEqual(result["unresolved_candidates"], 1)
            self.assertEqual(result["time_to_target_s"], 5)

    def test_continuation_segments_form_one_trajectory_without_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = build(
                Path(tmp),
                {
                    1: [
                        dict(rollout=1, start=0, end=4, finish="length", candidates=[]),
                        dict(rollout=2, start=10, end=16, finish="stop", parent=1, candidates=[("70", 14, True)]),
                    ]
                },
                target=1,
            )
            (trajectory,) = data["trajectories"]
            self.assertEqual(trajectory["active"], 10)
            self.assertFalse(trajectory["censored"])
            # Recorded run keeps the barrier gap; a slot-bounded replay removes it.
            self.assertEqual(simulate.simulate(data)["time_to_target_s"], 17)
            self.assertEqual(simulate.simulate(data, slots=1)["time_to_target_s"], 3 + 8)


class ExampleAttemptTests(unittest.TestCase):
    def test_reproduces_recorded_time_to_target(self):
        data = simulate.load_trajectories(EXAMPLE)
        recorded = data["recorded_solved_s"][17]
        replayed = simulate.simulate(data)
        self.assertAlmostEqual(replayed["time_to_target_s"], recorded, delta=0.05)
        self.assertEqual((replayed["checks"], replayed["wrong_checks"]), (18, 0))
        self.assertEqual(len(data["trajectories"]), 30)

    def test_counterfactual_ordering_is_consistent(self):
        data = simulate.load_trajectories(EXAMPLE)
        early = simulate.simulate(data, slots=30)["time_to_target_s"]
        bound = simulate.simulate(data, slots=30, extraction="final", censored="lower_bound")["time_to_target_s"]
        # Final-only cannot beat early extraction; the lower bound already shows it is slower.
        self.assertLess(early, bound)
        self.assertGreater(simulate.simulate(data, slots=10)["time_to_target_s"], early)

    def test_cli_compare_and_json(self):
        out = io.StringIO()
        with redirect_stdout(out):
            simulate.main([str(EXAMPLE), "--compare"])
        self.assertIn("recorded policy (reproduction check)", out.getvalue())
        out = io.StringIO()
        with redirect_stdout(out):
            simulate.main([str(EXAMPLE), "--slots", "30", "--json"])
        self.assertIn("time_to_target_s", json.loads(out.getvalue())["runs"][0])


if __name__ == "__main__":
    unittest.main()
