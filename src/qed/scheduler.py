"""V1 bounded rounds and independent continuation lanes."""

import asyncio
from copy import copy
from qed.lib.storage import AttemptArtifacts
from qed.lib.continuations import continuation_prefix
from qed.question import run_question


def question_state(output, index, artifacts=None):
    path = output / "trace" / f"{index:02d}" / "question.json"
    artifacts = artifacts or AttemptArtifacts(output)
    return artifacts.read_json(path) if artifacts.has_json(path) else None


def group_options(problem, args, output, artifacts=None):
    """Reserve at most the remaining budget, continuing each distinct lane once."""
    previous = question_state(output, problem["problem_idx"], artifacts)
    used = len(previous["rollouts"]) if previous else 0
    options = copy(args)
    options.rollouts = min(args.rollouts, args.max_attempts_per_question - used)
    options.max_tokens = args.max_tokens if used else args.first_pass_max_tokens
    round_no = previous["round"] + 1 if previous else 1
    prefixes = [None] * options.rollouts
    if previous and (not args.no_continuation):
        folder = output / "trace" / f"{problem['problem_idx']:02d}"
        last_group = [
            r for r in previous["rollouts"] if r["round"] == previous["round"]
        ]
        prefixes = [
            continuation_prefix(
                {"rollouts": [r]}, folder, args.max_context_tokens, artifacts
            )
            for r in last_group[: options.rollouts]
        ]
    return (options, round_no, used, prefixes)


class RoundScheduler:
    """FIFO question groups; default barriers include pending grader checks.

    A positive verdict is timestamped by QuestionRun before cancellation. The
    target event is signalled after that question's streams settle, as in v1.
    """

    def __init__(
        self, problems, args, client, output, sampler, attempt_start, profiler=None
    ):
        self.problems, self.args, self.client = problems, args, client
        self.output, self.sampler, self.attempt_start = output, sampler, attempt_start
        self.profiler = profiler
        self.artifacts = profiler.artifacts if profiler else AttemptArtifacts(output)
        self.target, self.solved = asyncio.Event(), set()

    def on_solved(self, event):
        self.solved.add(event["problem_idx"])
        if len(self.solved) >= self.args.target_correct:
            self.target.set()

    def eligible(self, problem):
        row = question_state(self.output, problem["problem_idx"], self.artifacts)
        return not row or (
            row["status"] == "unsolved"
            and len(row["rollouts"]) < self.args.max_attempts_per_question
            and row["round"] < self.args.max_rounds
        )

    async def worker(self, queue, retry):
        while not self.target.is_set():
            problem = await queue.get()
            try:
                options, round_no, used, prefixes = group_options(
                    problem, self.args, self.output, self.artifacts
                )
                row = await run_question(
                    problem,
                    options,
                    self.client,
                    self.output,
                    self.sampler,
                    round_no=round_no,
                    rollout_offset=used,
                    attempt_start=self.attempt_start,
                    on_solved=self.on_solved,
                    target_event=self.target,
                    profiler=self.profiler,
                    continuations=prefixes,
                )
                if row["status"] == "error":
                    raise RuntimeError(
                        f"Q{problem['problem_idx']} failed; inspect its trace"
                    )
                if retry and not self.target.is_set() and self.eligible(problem):
                    queue.put_nowait(problem)
            finally:
                queue.task_done()

    async def batch(self, pending, retry):
        queue = asyncio.Queue()
        for problem in pending:
            queue.put_nowait(problem)
        workers = [
            asyncio.create_task(self.worker(queue, retry))
            for _ in range(min(self.args.parallelism, len(pending)))
        ]
        all_workers = asyncio.gather(*workers)
        drained = asyncio.create_task(queue.join())
        try:
            await asyncio.wait(
                [all_workers, drained], return_when=asyncio.FIRST_COMPLETED
            )
            if all_workers.done():
                await all_workers
        finally:
            drained.cancel()
            for task in workers:
                if not task.done() and not task.cancelling():
                    task.cancel()
            await asyncio.gather(*workers, drained, return_exceptions=True)
            await asyncio.gather(all_workers, return_exceptions=True)

    async def run(self):
        for _ in range(self.args.max_rounds):
            pending = [p for p in self.problems if self.eligible(p)]
            if not pending or self.target.is_set():
                break
            work = asyncio.create_task(
                self.batch(pending, retry=self.args.schedule == "eager")
            )
            reached = asyncio.create_task(self.target.wait())
            try:
                await asyncio.wait([work, reached], return_when=asyncio.FIRST_COMPLETED)
                if self.target.is_set():
                    work.cancel()
                    await asyncio.gather(work, return_exceptions=True)
                    break
                await work
            finally:
                reached.cancel()
                if not work.done() and not work.cancelling():
                    work.cancel()
                await asyncio.gather(work, reached, return_exceptions=True)
            if self.args.schedule == "eager":
                break
        return self.artifacts.questions()


async def run_speedrun(
    problems, args, client, output, sampler, attempt_start, profiler=None
):
    return await RoundScheduler(
        problems, args, client, output, sampler, attempt_start, profiler
    ).run()
