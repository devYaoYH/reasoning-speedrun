"""Streaming v1 candidate policy with separately bounded fresh trajectories."""

import asyncio
from copy import copy
from qed.lib.generation import RolloutGenerator
from qed.question import QuestionRun
from .allocation import AllocationPool
from .budgets import first_plan, followup, request_bound


class PoolQuestion(QuestionRun):
    def __init__(self, problem, args, client, output, sampler, *, pool, **kwargs):
        self.pool, self.policy_args = pool, args
        options = copy(args)
        options.max_attempts_per_question = request_bound(args)
        super().__init__(problem, options, client, output, sampler, **kwargs)
        self.streams = []
        self.exhausted = asyncio.Event()

    def on_exhausted(self):
        self.candidates.put_nowait(None)
        self.exhausted.set()

    def coverage_finished(self):
        # This sentinel follows every candidate from the initial generation.
        # The shared verifier drains them before acknowledging the barrier.
        self.candidates.put_nowait(None)

    async def verify_candidates(self):
        async with asyncio.timeout(self.args.question_timeout):
            await super().verify_candidates()
            if self.winner:
                return
            self.pool.coverage_settled(self.index)
            await self.pool.expanded.wait()
            await super().verify_candidates()

    async def generate(self, rollout, plan):
        self.streams.append(asyncio.current_task())
        proxy = copy(self)
        proxy.args = copy(self.args)
        proxy.args.max_tokens = plan.max_tokens
        proxy.continuations = [plan.continuation]
        proxy.rollout_offset = rollout - 1
        # Preserve the v1 fresh-sample seed mapping. Disjoint continuation seed
        # bands make seeds independent of sibling admission/termination order.
        wanted = (
            self.policy_args.seed
            + self.index * 4
            + plan.fresh_sample
            + (plan.segment - 1) * self.policy_args.segment_seed_stride
        )
        proxy.args.seed = (
            wanted - self.index * proxy.args.max_attempts_per_question - rollout
        )
        generator = RolloutGenerator(proxy, rollout)
        try:
            await generator.run()
        finally:
            if hasattr(generator, "record"):
                generator.record.update(
                    fresh_sample=plan.fresh_sample,
                    segment=plan.segment,
                    trajectory_generated_before=plan.generated_before,
                    trajectory_generated_tokens=plan.generated_before
                    + generator.record["generated_token_ids_count"],
                )
                self.write_json(generator.path / "telemetry.json", generator.record)
        return followup(
            plan, generator.record, self.policy_args, self.folder, self.artifacts
        )

    async def run(self):
        self.pool.register(
            self.index,
            self.generate,
            lambda sample: first_plan(self.problem, self.policy_args, sample),
            self.on_exhausted,
            self.coverage_finished,
        )
        self.producer = asyncio.create_task(self.exhausted.wait())
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
            await self.pool.close_question(self.index)
            self.exhausted.set()
            await self.settle()
            self.pool.coverage_settled(self.index)
        if self.result["status"] == "error":
            raise RuntimeError(f"Q{self.index} failed: {self.question_error}")
        return self.result


async def run_speedrun(
    problems, args, client, output, sampler, attempt_start, profiler
):
    args.segment_seed_stride = (max(p["problem_idx"] for p in problems) + 1) * 4
    pool = AllocationPool(
        [p["problem_idx"] for p in problems],
        args,
        profiler.artifacts,
        output,
        attempt_start,
    )
    target, solved = asyncio.Event(), set()

    def on_solved(event):
        solved.add(event["problem_idx"])
        if len(solved) >= args.target_correct:
            target.set()
            pool.halt_requested = True
            pool.changed.set()

    tasks = {
        p["problem_idx"]: asyncio.create_task(
            PoolQuestion(
                p,
                args,
                client,
                output,
                sampler,
                pool=pool,
                attempt_start=attempt_start,
                target_event=target,
                on_solved=on_solved,
                profiler=profiler,
            ).run()
        )
        for p in problems
    }
    work = asyncio.gather(*tasks.values())
    allocator, reached = (
        asyncio.create_task(pool.run()),
        asyncio.create_task(target.wait()),
    )
    try:
        await asyncio.wait(
            [work, allocator, reached], return_when=asyncio.FIRST_COMPLETED
        )
        if allocator.done():
            await allocator
            await asyncio.wait([work, reached], return_when=asyncio.FIRST_COMPLETED)
        if work.done():
            rows = await work
            if any(row["status"] == "error" for row in rows):
                raise RuntimeError(
                    "Question failed; inspect verification and generation traces"
                )
    finally:
        for index, task in tasks.items():
            if index not in solved and not task.done() and not task.cancelling():
                task.cancel()
        for task in (allocator, reached):
            if not task.done() and not task.cancelling():
                task.cancel()
        await asyncio.gather(
            *tasks.values(), work, allocator, reached, return_exceptions=True
        )
        pool.snapshot()
    return profiler.artifacts.questions()
