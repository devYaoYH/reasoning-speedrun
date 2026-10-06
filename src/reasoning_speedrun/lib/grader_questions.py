"""Read the grader's question set without local dataset or answer-key loading."""

import hashlib
import json


def question_digest(questions):
    raw = json.dumps(
        questions, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(raw).hexdigest()


async def fetch_questions(client, args, health):
    response = await client.get(args.grader_url + "/questions")
    response.raise_for_status()
    payload = response.json()
    questions = payload.get("questions")
    if not isinstance(questions, list) or not questions:
        raise RuntimeError("Grader did not provide a nonempty question set")
    for row in questions:
        if (
            not isinstance(row, dict)
            or set(row) != {"problem_idx", "problem"}
            or type(row["problem_idx"]) is not int
            or row["problem_idx"] < 1
            or not isinstance(row["problem"], str)
            or not row["problem"].strip()
        ):
            raise RuntimeError(
                "Grader questions must contain only positive indices and problem statements"
            )
    indices = [q["problem_idx"] for q in questions]
    if indices != sorted(set(indices)):
        raise RuntimeError("Grader question indices must be ordered and unique")
    digest = question_digest(questions)
    dataset = payload.get("dataset", {})
    if (
        digest != payload.get("questions_sha256")
        or digest != health.get("dataset", {}).get("questions_sha256")
        or dataset.get("grader_sha256") != health.get("dataset", {}).get("sha256")
        or dataset.get("rows") != len(questions)
        or health.get("n_problems") != len(questions)
    ):
        raise RuntimeError(
            "Grader question set disagrees with its health/provenance fingerprint"
        )
    if args.questions:
        wanted = set(args.questions)
        if not wanted <= set(indices):
            raise ValueError("Unknown question index in grader question set")
        selected = [q for q in questions if q["problem_idx"] in wanted]
    else:
        selected = questions
    if args.target_correct > len(selected):
        raise ValueError("Target correct exceeds the number of selected questions")
    dataset = {**dataset, "role": args.benchmark_role}
    return selected, questions, dataset, digest
