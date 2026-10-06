"""A question group: propose, deduplicate, verify, cancel and persist."""

import asyncio
import time
from qed.lib.storage import AttemptArtifacts
from qed.lib.common import utc_now
from qed.lib.metrics import Meter, merge_meters
from qed.lib.continuations import continuation_prefix
from qed.lib.generation import RolloutGenerator


class QuestionRun:
    # Policies swap the per-stream generator (e.g. to change answer extraction).
    generator = RolloutGenerator

    def __init__(
        self,
        problem,
        args,
        client,
        output,
        sampler,
        *,
        round_no=1,
        rollout_offset=0,
        attempt_start=None,
        on_solved=None,
        target_event=None,
        profiler=None,
        continuations=None,
    ):
        self.problem = problem
        self.args = args
        self.client = client
        self.output = output
        self.sampler = sampler
        self.round_no = round_no
        self.rollout_offset = rollout_offset
        self.attempt_start = attempt_start
        self.on_solved = on_solved
        self.target_event = target_event
        self.profiler = profiler
        self.continuations = continuations
        self.meter = Meter(
            not getattr(self.args, "no_overhead_profile", False),
            parent=self.profiler.meter if self.profiler else None,
        )
        self.rollout_meters = {}
        self.artifacts = (
            self.profiler.artifacts if self.profiler else AttemptArtifacts(self.output)
        )
        self.index = self.problem["problem_idx"]
        self.folder = self.output / "trace" / f"{self.index:02d}"
        if not self.artifacts.buffered:
            self.folder.mkdir(parents=True, exist_ok=True)
        self.previous = (
            self.artifacts.read_json(self.folder / "question.json")
            if self.artifacts.has_json(self.folder / "question.json")
            else None
        )
        if self.previous and self.previous["status"] == "solved":
            raise ValueError("Solved questions must not be retried")
        if (
            self.previous
            and len(self.previous["rollouts"]) + self.args.rollouts
            > self.args.max_attempts_per_question
        ):
            raise ValueError("Per-question generation attempt limit exceeded")
        self.next_continuation = None
        if self.continuations is None and (not self.args.no_continuation):
            self.next_continuation = continuation_prefix(
                self.previous, self.folder, self.args.max_context_tokens, self.artifacts
            )
        self.start, self.started = (time.perf_counter(), utc_now())
        self.candidates = asyncio.Queue()
        self.seen = (
            set(self.previous.get("candidate_answers", [])) if self.previous else set()
        )
        self.records = []
        self.winner = None
        self.question_error = None
        self.solved_event = None
        self.stopped_for_target = False

    def write_json(self, path, value, scope=None):
        scope = scope or self.meter
        with scope.measure("artifact_json_write"):
            self.artifacts.write_json(path, value)

    def propose(self, event, rollout):
        candidate = str(event["answer"])
        self.meter.inc("candidate_proposals")
        if candidate not in self.seen:
            self.seen.add(candidate)
            self.candidates.put_nowait(
                {
                    **event,
                    "candidate": candidate,
                    "rollout": rollout,
                    "observed_at_utc": utc_now(),
                    "question_elapsed_s": time.perf_counter() - self.start,
                    "_enqueued_monotonic_s": time.perf_counter(),
                }
            )
            self.meter.inc("candidate_unique_enqueued")
            self.meter.high_water("candidate_queue_depth", self.candidates.qsize())
        else:
            self.meter.inc("candidate_duplicates_suppressed")

    async def drained(self):
        try:
            await asyncio.gather(*self.streams)
        finally:
            self.candidates.put_nowait(None)

    async def verify_candidates(self):
        async with asyncio.timeout(self.args.question_timeout):
            with self.artifacts.open_jsonl(self.folder / "verification.jsonl") as file:
                while True:
                    event = await self.candidates.get()
                    if event is None:
                        break
                    queue_wait = time.perf_counter() - event.pop(
                        "_enqueued_monotonic_s"
                    )
                    self.meter.observe("candidate_local_queue_wait", queue_wait)
                    event.update(
                        verification_started_at_utc=utc_now(),
                        round=self.round_no,
                        candidate_queue_wait_s=queue_wait,
                    )
                    verify_start = time.perf_counter()
                    self.meter.inc("verification_submitted")
                    if self.profiler:
                        self.profiler.submitted(
                            verify_start
                            - (
                                self.attempt_start
                                if self.attempt_start is not None
                                else self.start
                            )
                        )
                    try:
                        response = await self.client.post(
                            self.args.grader_url + "/verify",
                            json={
                                "index": self.index,
                                "candidate": event["candidate"],
                                "agent_id": f"{self.output.name}-q{self.index}-r{event['rollout']}",
                                "query_id": f"{self.output.name}-q{self.index}-a{event['candidate']}",
                            },
                        )
                        response.raise_for_status()
                        verdict = response.json()
                        if type(verdict.get("verdict")) is not bool:
                            raise RuntimeError(
                                "Grader response lacks a boolean verdict"
                            )
                        event["result"] = verdict
                        self.meter.inc("verification_completed")
                        self.meter.inc(
                            "verification_correct"
                            if verdict["verdict"]
                            else "verification_wrong"
                        )
                        if verdict.get("queue_wait_s") is not None:
                            self.meter.observe(
                                "grader_queue_wait", verdict["queue_wait_s"]
                            )
                        if verdict.get("toll_s") is not None:
                            self.meter.observe("grader_service", verdict["toll_s"])
                    except asyncio.CancelledError:
                        event["cancelled"] = True
                        self.meter.inc("verification_cancelled")
                        raise
                    except Exception as exc:
                        self.meter.inc("verification_errors")
                        event["error"] = f"{type(exc).__name__}: {exc}"
                        raise
                    finally:
                        event.update(
                            verification_finished_at_utc=utc_now(),
                            verification_latency_s=time.perf_counter() - verify_start,
                        )
                        self.meter.observe(
                            "verification_http_wait", event["verification_latency_s"]
                        )
                        with self.meter.measure("verification_trace_write_flush"):
                            file.append(event)
                    if verdict["verdict"]:
                        self.winner = event
                        self.solved_event = {
                            "problem_idx": self.index,
                            "round": self.round_no,
                            "rollout": event["rollout"],
                            "candidate": event["candidate"],
                            "first_solved_at_utc": utc_now(),
                            "first_solved_elapsed_s": time.perf_counter()
                            - (
                                self.attempt_start
                                if self.attempt_start is not None
                                else self.start
                            ),
                            "grader_answered_at_utc": verdict.get("answered_at"),
                            "grader_query_id": verdict.get("query_id"),
                        }
                        with self.artifacts.open_jsonl(
                            self.output / "solved.jsonl"
                        ) as solved_file:
                            solved_file.append(self.solved_event)
                        break

    async def settle(self):
        cancellation_start = time.perf_counter()
        for task in self.streams:
            if not task.done() and (not task.cancelling()):
                task.cancel()
        await asyncio.gather(*self.streams, return_exceptions=True)
        await asyncio.gather(self.producer, return_exceptions=True)
        self.meter.observe(
            "generation_cancellation_settlement",
            time.perf_counter() - cancellation_start,
        )
        ended, ended_at = (time.perf_counter(), utc_now())
        for path, record, _ in self.records:
            record.update(
                finished_at_utc=ended_at,
                end_to_end_latency_s=ended - record["start_monotonic_s"],
                end_to_end_scope="rollout start through question verification/cancellation settlement",
                gpu=None,
            )
            with self.meter.measure("gpu_sample_window"):
                record["gpu"] = self.sampler.window(
                    record["start_monotonic_s"], record["generation_end_monotonic_s"]
                )
            record["overhead"] = self.rollout_meters[record["rollout"]].snapshot()
            self.write_json(path / "telemetry.json", record)
        self.result = {
            "problem_idx": self.index,
            "started_at_utc": self.started,
            "finished_at_utc": ended_at,
            "end_to_end_latency_s": ended - self.start,
            "status": (
                "error"
                if self.question_error
                else (
                    "solved"
                    if self.winner
                    else "stopped" if self.stopped_for_target else "unsolved"
                )
            ),
            "winner": self.winner,
            "error": self.question_error,
            "unique_candidates": len(self.seen),
            "candidate_answers": sorted(self.seen),
            "rollouts": [
                record
                for _, record, _ in sorted(self.records, key=lambda r: r[1]["rollout"])
            ],
        }
        if any((r["status"] == "error" for r in self.result["rollouts"])) and (
            not self.winner
        ):
            self.result["status"] = "error"
        self.result.update(
            round=self.round_no,
            first_solved=self.solved_event,
            overhead=self.meter.snapshot(),
        )
        self.write_json(self.folder / f"round-{self.round_no:02d}.json", self.result)
        rounds = (self.previous.get("rounds", []) if self.previous else []) + [
            {
                k: self.result[k]
                for k in (
                    "round",
                    "status",
                    "started_at_utc",
                    "finished_at_utc",
                    "end_to_end_latency_s",
                    "winner",
                    "error",
                )
            }
        ]
        if self.previous:
            self.result["rollouts"] = (
                self.previous["rollouts"] + self.result["rollouts"]
            )
            self.result["started_at_utc"] = self.previous["started_at_utc"]
            self.result["first_solved"] = (
                self.previous.get("first_solved") or self.solved_event
            )
        self.result["overhead"] = merge_meters(
            self.previous.get("overhead") if self.previous else None,
            self.meter.snapshot(),
        )
        self.result["rounds"] = rounds
        self.result["question_start_monotonic_s"] = (
            self.previous.get("question_start_monotonic_s", self.start)
            if self.previous
            else self.start
        )
        self.result["end_to_end_latency_s"] = (
            ended - self.result["question_start_monotonic_s"]
        )
        self.write_json(self.folder / "question.json", self.result)
        if self.solved_event and self.on_solved:
            self.on_solved(self.solved_event)
        print(
            f"Q{self.index:02d} round {self.round_no}: {self.result['status']}, {ended - self.start:.2f}s",
            flush=True,
        )

    async def run(self):
        self.streams = [
            asyncio.create_task(self.generator(self, self.rollout_offset + r).run())
            for r in range(1, self.args.rollouts + 1)
        ]
        self.producer = asyncio.create_task(self.drained())
        try:
            await self.verify_candidates()
        except asyncio.CancelledError:
            self.stopped_for_target = (
                self.target_event is not None and self.target_event.is_set()
            )
            self.question_error = (
                None if self.stopped_for_target else "attempt interrupted"
            )
            raise
        except Exception as exc:
            self.question_error = f"{type(exc).__name__}: {exc}"
        finally:
            await self.settle()
        return self.result


async def run_question(
    problem,
    args,
    client,
    output,
    sampler,
    *,
    round_no=1,
    rollout_offset=0,
    attempt_start=None,
    on_solved=None,
    target_event=None,
    profiler=None,
    continuations=None,
):
    return await QuestionRun(
        problem,
        args,
        client,
        output,
        sampler,
        round_no=round_no,
        rollout_offset=rollout_offset,
        attempt_start=attempt_start,
        on_solved=on_solved,
        target_event=target_event,
        profiler=profiler,
        continuations=continuations,
    ).run()
