"""Dual-mode grader entry point built on the vendored MathArena parser.

grade(candidate, gold) -> bool
  True iff `candidate` is mathematically equivalent to `gold`.
  `candidate` may be a bare answer ("204", "\\frac{1}{2}") OR full model text
  containing "\\boxed{...}". `gold` is the dataset's answer string.
Robust: any parse/compare failure -> False (never raises).
"""
from parser import extract_answer, parse_answer, check_answers

# Silence the upstream parser's expected loguru WARNING spam on exotic answers.
try:
    from loguru import logger as _lg
    _lg.disable("parser")
    _lg.disable("parse_manual")
except Exception:
    pass


def _parse_candidate(candidate: str):
    # If the agent submitted full text with a box, extract from the box; otherwise
    # treat the whole string as the answer.
    if "boxed" in candidate or "fbox" in candidate:
        ans, _ = extract_answer(candidate, parse=True)
        return ans
    ans, _ = parse_answer(str(candidate))
    return ans


def grade(candidate, gold) -> bool:
    if candidate is None or gold is None:
        return False
    try:
        cand = _parse_candidate(str(candidate))
    except Exception:
        return False
    if cand is None:
        return False
    try:
        gold_ans, _ = parse_answer(str(gold))
    except Exception:
        return False
    try:
        return bool(check_answers(cand, gold_ans))
    except Exception:
        return False
