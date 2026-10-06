"""Initial coverage barrier, then a rotating pool with a fresh-sample cap."""

import asyncio
from collections import deque
import time
from qed.lib.common import utc_now


class AllocationPool:
    def __init__(self, indices, args, artifacts, output, attempt_start):
        self.args, self.artifacts, self.output = args, artifacts, output
        self.attempt_start = attempt_start
        self.states = {
            q: dict(
                fresh=0,
                requests=0,
                active=0,
                peak_active=0,
                closed=False,
                exhausted=False,
                ready=deque(),
                coverage_settled=False,
                coverage_settled_elapsed_s=None,
                initial_generation_retired_elapsed_s=None,
            )
            for q in indices
        }
        self.tasks = {}
        self.changed, self.registered = asyncio.Event(), asyncio.Event()
        self.halt_requested, self.cursor, self.peak_active = False, 0, 0
        self.admissions = []
        self.phase = "coverage"
        self.expanded = asyncio.Event()
        self.barrier_release = None

    def register(self, index, generate, fresh_plan, exhausted, coverage_finished):
        self.states[index].update(
            generate=generate,
            fresh_plan=fresh_plan,
            on_exhausted=exhausted,
            coverage_finished=coverage_finished,
        )
        if all("generate" in s for s in self.states.values()):
            self.registered.set()

    def retire(self):
        for task, index in list(self.tasks.items()):
            if not task.done():
                continue
            del self.tasks[task]
            state = self.states[index]
            state["active"] -= 1
            if self.phase == "coverage":
                state["initial_generation_retired_elapsed_s"] = (
                    time.perf_counter() - self.attempt_start
                )
            if task.cancelled():
                if not state["closed"]:
                    raise RuntimeError(
                        f"Unexpected generation cancellation for Q{index}"
                    )
                continue
            next_plan = task.result()
            if next_plan and not state["closed"]:
                state["ready"].append(next_plan)
            if self.phase == "coverage" and not state["closed"]:
                state["coverage_finished"]()

    def coverage_settled(self, index):
        state = self.states[index]
        if not state["coverage_settled"]:
            state["coverage_settled"] = True
            state["coverage_settled_elapsed_s"] = (
                time.perf_counter() - self.attempt_start
            )
        self.changed.set()

    def choose(self):
        indices = list(self.states)
        ordered = indices[self.cursor :] + indices[: self.cursor]
        eligible = [
            q
            for q in ordered
            if not self.states[q]["closed"]
            and not self.states[q]["exhausted"]
            and (
                self.states[q]["ready"]
                or self.states[q]["fresh"] < self.args.max_fresh_samples_per_question
            )
        ]
        if not eligible or self.halt_requested:
            return None
        index = min(
            eligible,
            key=lambda q: (not self.states[q]["ready"], self.states[q]["active"]),
        )
        self.cursor = (indices.index(index) + 1) % len(indices)
        return index

    def admit(self, index):
        state = self.states[index]
        if state["ready"]:
            plan = state["ready"].popleft()
        else:
            if state["fresh"] >= self.args.max_fresh_samples_per_question:
                raise RuntimeError("Fresh sample cap exceeded")
            state["fresh"] += 1
            plan = state["fresh_plan"](state["fresh"])
        state["requests"] += 1
        task = asyncio.create_task(state["generate"](state["requests"], plan))
        self.tasks[task] = index
        state["active"] += 1
        state["peak_active"] = max(state["peak_active"], state["active"])
        self.peak_active = max(self.peak_active, len(self.tasks))
        self.admissions.append(
            dict(
                problem_idx=index,
                phase=self.phase,
                rollout=state["requests"],
                fresh_sample=plan.fresh_sample,
                segment=plan.segment,
                parent_rollout=(plan.continuation or {}).get("parent_rollout"),
                max_tokens=plan.max_tokens,
                generated_before=plan.generated_before,
                released_at_utc=utc_now(),
                elapsed_s=time.perf_counter() - self.attempt_start,
                active_requests=len(self.tasks),
                question_active_requests=state["active"],
            )
        )
        task.add_done_callback(lambda _: self.changed.set())

    async def close_question(self, index):
        state = self.states[index]
        state["closed"] = True
        state["ready"].clear()
        tasks = [t for t, q in self.tasks.items() if q == index]
        for task in tasks:
            if not task.done() and not task.cancelling():
                task.cancel()
        self.changed.set()
        await asyncio.gather(*tasks, return_exceptions=True)

    def snapshot(self):
        self.artifacts.write_json(
            self.output / "allocation.json",
            dict(
                policy="initial one-per-question coverage barrier including queued checks; then continuations first, least-active fresh samples with rotating ties",
                barrier_release=self.barrier_release,
                max_concurrent_requests=self.args.max_concurrent_requests,
                peak_active_requests=self.peak_active,
                max_fresh_samples_per_question=self.args.max_fresh_samples_per_question,
                max_rollout_tokens=self.args.max_rollout_tokens,
                questions={
                    str(q): {
                        k: s[k]
                        for k in (
                            "fresh",
                            "requests",
                            "active",
                            "peak_active",
                            "closed",
                            "exhausted",
                            "coverage_settled",
                            "coverage_settled_elapsed_s",
                            "initial_generation_retired_elapsed_s",
                        )
                    }
                    for q, s in self.states.items()
                },
                admissions=self.admissions,
            ),
        )

    async def run(self):
        await self.registered.wait()
        try:
            # Admit exactly one initial sample per question, even with a larger
            # configured pool. Freed slots remain idle until coverage settles.
            for index in self.states:
                self.admit(index)
            while True:
                self.changed.clear()
                self.retire()
                if self.halt_requested:
                    return
                if not self.tasks and all(
                    s["coverage_settled"] for s in self.states.values()
                ):
                    break
                await self.changed.wait()
            self.phase = "pool"
            self.barrier_release = dict(
                released_at_utc=utc_now(),
                elapsed_s=time.perf_counter() - self.attempt_start,
                active_requests=0,
            )
            self.expanded.set()
            while True:
                self.changed.clear()
                self.retire()
                if self.halt_requested:
                    break
                for state in self.states.values():
                    if (
                        not state["closed"]
                        and not state["exhausted"]
                        and not state["active"]
                        and not state["ready"]
                        and state["fresh"] >= self.args.max_fresh_samples_per_question
                    ):
                        state["exhausted"] = True
                        state["on_exhausted"]()
                while len(self.tasks) < self.args.max_concurrent_requests:
                    index = self.choose()
                    if index is None:
                        break
                    self.admit(index)
                if not self.tasks and all(
                    s["closed"] or s["exhausted"] for s in self.states.values()
                ):
                    break
                await self.changed.wait()
        finally:
            for state in self.states.values():
                state["closed"] = True
            for task in self.tasks:
                if not task.done() and not task.cancelling():
                    task.cancel()
            await asyncio.gather(*self.tasks, return_exceptions=True)
            self.tasks.clear()
            for state in self.states.values():
                state["active"] = 0
            self.snapshot()
