"""Complete fan-out, no intermediate extraction.

Every sample of every question starts at once (concurrency is bounded only by
the inference server's own scheduler). A sample contributes a candidate only
after it ends naturally, from the last closed integer box of its final response.
Nothing is read from the reasoning stream, so there is no early exit inside a
trajectory. A verified-correct answer still cancels that question's remaining
samples, and reaching the solve target ends the attempt.
"""

import asyncio
import re

from reasoning_speedrun.lib.generation import RolloutGenerator
from reasoning_speedrun.lib.storage import AttemptArtifacts
from reasoning_speedrun.question import QuestionRun

BOX = re.compile(r"\\boxed\s*\{\s*(\d{1,3})\s*\}")
OPEN_BOX = re.compile(r"\\boxed\s*\{")


def final_answer(text):
    """The last box of a naturally ended final response, if it is an integer 0-999."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    elif "<think>" in text:
        return None
    opened = list(OPEN_BOX.finditer(text))
    match = BOX.match(text[opened[-1].start() :]) if opened else None
    if match is None:
        return None
    return {
        "answer": int(match[1]),
        "part": "content",
        "kind": "final_box",
        "extraction_stage": "completed_final_response",
    }


class FinalOnlyGenerator(RolloutGenerator):
    """Ignore the stream; propose once, at natural end of the content channel."""

    def detect(self, part, value, eof=False):
        if not eof or part != "content":
            return
        event = final_answer("".join(self.parts["content"]))
        if event:
            self.question.propose(event, self.rollout)


class NaiveQuestion(QuestionRun):
    generator = FinalOnlyGenerator


async def run_speedrun(
    problems, args, client, output, sampler, attempt_start, profiler=None
):
    target, solved = asyncio.Event(), set()

    def on_solved(event):
        solved.add(event["problem_idx"])
        if len(solved) >= args.target_correct:
            target.set()

    tasks = [
        asyncio.create_task(
            NaiveQuestion(
                problem,
                args,
                client,
                output,
                sampler,
                attempt_start=attempt_start,
                on_solved=on_solved,
                target_event=target,
                profiler=profiler,
            ).run()
        )
        for problem in problems
    ]
    finished = asyncio.gather(*tasks, return_exceptions=True)
    reached = asyncio.create_task(target.wait())
    try:
        await asyncio.wait([finished, reached], return_when=asyncio.FIRST_COMPLETED)
    finally:
        reached.cancel()
        for task in tasks:
            if not task.done() and not task.cancelling():
                task.cancel()
        await asyncio.gather(finished, reached, return_exceptions=True)
    artifacts = profiler.artifacts if profiler else AttemptArtifacts(output)
    return artifacts.questions()
