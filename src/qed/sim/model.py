"""Fake tokenizer and the scripted "model": deterministic trajectories with planted answers.

A trajectory is a token sequence: filler reasoning with one or two answer
statements planted at chosen positions, then a short final response. Answers
use the same surface forms the real extractor recognises (a closed box, an
``Answer:`` line, an "answer is N" clause), so early extraction, deduplication,
wrong-answer costs and continuations all exercise the real code paths.
"""

from dataclasses import dataclass, field
import math
import random
import re
import zlib

PIECE = re.compile(r"\s*[A-Za-z]+|\s*\d+|\s*[^\sA-Za-z\d]|\s+")
FILLER = (
    "so then let us compute the sum of a and b modulo n hmm wait consider the case "
    "where x is an integer divisor check again therefore note that if and only if "
    "we get the equation holds try verify count residue bound expand simplify"
).split()


def token_id(piece):
    return 1 + zlib.crc32(piece.strip().encode() or b" ") % 150_000


def pieces(text):
    return PIECE.findall(text)


def encode(text):
    """Deterministic stand-in tokenizer: one token per word, number or symbol."""
    return [token_id(p) for p in pieces(text)]


def chat_prompt_ids(messages):
    ids = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, list):
            content = " ".join(part.get("text", "") for part in content if isinstance(part, dict))
        ids += encode(f"<{message.get('role', 'user')}>") + encode(content)
    return ids + encode("<assistant>")


@dataclass
class Plan:
    """One planned trajectory; ``render`` produces tokens lazily so long ones stay cheap."""

    reasoning: int
    salt: int
    inserts: dict = field(default_factory=dict)  # position -> (text, id)
    final_tokens: list = field(default_factory=list)  # (text, id) after the reasoning
    final_answer: int = 0
    gold: int | None = None
    kinds: list = field(default_factory=list)

    @property
    def total(self):
        return self.reasoning + len(self.final_tokens)

    def token(self, i):
        if i >= self.reasoning:
            part, (text, tid) = "content", self.final_tokens[i - self.reasoning]
        elif i in self.inserts:
            part, (text, tid) = "reasoning", self.inserts[i]
        else:
            word = FILLER[(i * 7 + self.salt) % len(FILLER)]
            part, (text, tid) = "reasoning", (" " + word, token_id(word))
        return part, text, tid

    def render(self, start, end):
        return [self.token(i) for i in range(start, min(end, self.total))]


def statement(rng, answer):
    """An answer statement in one of the forms the extractor recognises."""
    kind = rng.choice(("boxed", "prose", "line"))
    text = {
        "boxed": f" So the result is \\boxed{{{answer}}}.",
        "prose": f" Thus the answer is {answer}.",
        "line": f"\nAnswer: {answer}\n",
    }[kind]
    return kind, text


def wrong_answer(rng, avoid):
    while True:
        value = rng.randrange(0, 1000)
        if value not in avoid:
            return value


def build_plan(behavior, rng, gold, min_tokens=0):
    """Plan a trajectory; ``gold`` is None for prompts outside the answer key."""
    r = max(
        behavior.reasoning_min_tokens,
        int(rng.lognormvariate(math.log(behavior.reasoning_median_tokens), behavior.reasoning_sigma)),
    )
    correct = gold is not None and rng.random() < behavior.p_correct
    final = gold if correct else wrong_answer(rng, {gold})
    lo, hi = behavior.answer_at
    position = int(r * rng.uniform(lo, hi))
    statements = []
    if rng.random() < behavior.p_wrong_first:
        tentative = wrong_answer(rng, {gold, final})
        statements.append(statement(rng, tentative))
        second = " Wait, let me recheck." + statement(rng, final)[1]
        statements.append((statement(rng, final)[0], second))
    else:
        statements.append(statement(rng, final))
    inserts, cursor, kinds = {}, position, []
    for index, (kind, text) in enumerate(statements):
        words = pieces(text)
        if index:
            cursor += 1 + int((r - cursor) * rng.uniform(0.2, 0.6))
        r = max(r, cursor + len(words))
        for offset, word in enumerate(words):
            inserts[cursor + offset] = (word, token_id(word))
        kinds.append(kind)
        cursor += len(words)
    r = max(r, min_tokens)
    summary = " ".join(rng.choice(FILLER) for _ in range(max(0, behavior.final_tokens - 8)))
    final_text = f"{summary}\n\nFinal answer: \\boxed{{{final}}}"
    final_tokens = [(p, token_id(p)) for p in pieces(final_text)]
    return Plan(r, rng.randrange(len(FILLER)), inserts, final_tokens, final, gold, kinds)


def plan_rng(behavior_seed, request_seed, key):
    return random.Random(zlib.crc32(f"{behavior_seed}|{request_seed}|{key}".encode()))
