"""Read saved attempt artifacts for the local browser viewer.

Use AttemptStore with an attempts/ directory to inspect compact summaries or
locally copied trajectories, oracle events, and sampled device telemetry. Reads
never start inference or grading. Missing traces remain explicitly unavailable;
an incomplete final JSONL line is tolerated while a runner appends to the file.
"""
from __future__ import annotations

import json
from pathlib import Path
import re

from qed.lib.datasets import recorded_dataset

SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
ROLLOUT_FILES = {"request.json", "response.json", "telemetry.json", "tokens.json", "stream.jsonl"}
ATTEMPT_FILES = {"config.json", "summary.json", "metadata.json", "solved.jsonl", "gpu.jsonl", "questions.json"}


def dataset_record(config):
    return recorded_dataset(config)


def json_file(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def json_lines(path):
    if not path.is_file():
        return []
    text = path.read_text()
    lines = text.splitlines()
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1 or text.endswith("\n"):
                raise
    return rows


class AttemptStore:
    def __init__(self, root: Path):
        self.root = root

    def folder(self, name):
        if not SAFE_ID.fullmatch(name):
            raise ValueError("Invalid attempt identifier")
        path = self.root / name
        if not path.resolve().is_relative_to(self.root.resolve()) or not path.is_dir():
            raise FileNotFoundError(name)
        return path

    def path(self, folder, relative):
        path = folder / relative
        if not path.resolve().is_relative_to(folder.resolve()):
            raise ValueError("Artifact leaves its attempt directory")
        return path

    def read(self, folder, relative, default=None):
        return json_file(self.path(folder, relative), default)

    def lines(self, folder, relative):
        return json_lines(self.path(folder, relative))

    def metadata(self, name):
        folder = self.folder(name)
        config = self.read(folder, "config.json", {})
        summary = self.read(folder, "summary.json", {})
        if config.get("attempt_id") != name and summary.get("attempt_id") != name:
            raise ValueError("Not a canonical attempt")
        return {"id": name, "model": config.get("model"),
                "benchmark_id": dataset_record(config)["id"],
                "benchmark_year": dataset_record(config)["year"],
                "benchmark_role": dataset_record(config)["role"],
                "status": summary.get("status", "incomplete"),
                "solved": summary.get("solved"),
                "questions": len(config.get("question_indices") or config.get("questions") or []) or
                             (config.get("grader_health") or {}).get("n_problems"),
                "started_at_utc": summary.get("official_started_at_utc") or config.get("official_started_at_utc"),
                "official_latency_s": summary.get("official_latency_s"),
                "trace_questions": sum(self.path(folder, f"trace/{p.name}/question.json").is_file()
                                       for p in (folder / "trace").glob("[0-9]*") if p.is_dir())}

    def list(self):
        items, warnings = [], []
        for folder in sorted(self.root.iterdir(), reverse=True) if self.root.exists() else []:
            if not folder.is_dir() or not SAFE_ID.fullmatch(folder.name):
                continue
            if not (folder / "config.json").exists() and not (folder / "summary.json").exists():
                continue  # A runner or local copy may have just created the folder.
            try:
                items.append(self.metadata(folder.name))
            except (ValueError, OSError) as exc:
                warnings.append(f"{folder.name}: {exc}")
        return {"attempts": items, "warnings": warnings}

    def question(self, name, index):
        folder = self.folder(name)
        base = f"trace/{index:02d}"
        record = self.read(folder, f"{base}/question.json")
        rollouts = {int(r["rollout"]): r for r in (record or {}).get("rollouts", [])}
        for path in self.path(folder, base).glob("rollout-*"):
            if not re.fullmatch(r"rollout-\d+", path.name):
                continue
            number = int(path.name.split("-")[1])
            telemetry = self.read(folder, f"{base}/{path.name}/telemetry.json", {})
            rollouts[number] = {**rollouts.get(number, {}), **telemetry, "rollout": number}
        for number, rollout in rollouts.items():
            rollout["response_available"] = self.path(folder, f"{base}/rollout-{number:02d}/response.json").is_file()
        return {"problem_idx": index, "record": record,
                "trace_available": bool(record or rollouts),
                "rollouts": [rollouts[k] for k in sorted(rollouts)],
                "verification": self.lines(folder, f"{base}/verification.jsonl")}

    def overview(self, name):
        folder = self.folder(name)
        metadata = self.metadata(name)
        config = self.read(folder, "config.json", {})
        summary = self.read(folder, "summary.json")
        # Problem text comes from the attempt's own gold-free snapshot; without
        # one the viewer still shows every recorded trace, with blank statements.
        prompt_rows = json_file(folder / "questions.json", [])
        problems = {p["problem_idx"]: p["problem"] for p in prompt_rows}
        saved = {q["problem_idx"]: q for q in (summary or {}).get("questions", [])}
        indices = set(config.get("question_indices") or config.get("questions") or problems)
        indices.update(saved)
        indices.update(int(p.name) for p in self.path(folder, "trace").glob("[0-9]*") if p.name.isdigit())
        questions = []
        for index in sorted(indices):
            detail = self.question(name, index)
            record = detail["record"] or {}
            merged = {**saved.get(index, {}), **record}
            winner = merged.get("winner") or {}
            # Older canonical versions retained verdict timestamps in winner,
            # before first_solved was added. Preserve their measured event time.
            first_solved = merged.get("first_solved")
            if not first_solved and merged.get("status") == "solved" and winner.get("verification_finished_at_utc"):
                first_solved = {"problem_idx": index, "candidate": winner.get("candidate"),
                                "first_solved_at_utc": winner["verification_finished_at_utc"],
                                "source": "winner verification_finished_at_utc"}
            questions.append({"problem_idx": index, "problem": problems.get(index, ""),
                              "status": merged.get("status", "pending"),
                              "verified_answer": merged.get("verified_answer", winner.get("candidate")),
                              "winning_rollout": merged.get("winning_rollout", winner.get("rollout")),
                              "first_solved": first_solved,
                              "started_at_utc": merged.get("started_at_utc"),
                              "finished_at_utc": merged.get("finished_at_utc"),
                              "end_to_end_latency_s": merged.get("end_to_end_latency_s"),
                              "unique_candidates": merged.get("unique_candidates"),
                              "error": merged.get("error"), "rounds": merged.get("rounds", []),
                              "trace_available": detail["trace_available"], "rollouts": detail["rollouts"],
                              "verification_count": len(detail["verification"]) if detail["trace_available"] else None})
        events = self.lines(folder, "solved.jsonl")
        return {"attempt": metadata, "config": config, "summary": summary,
                "experiment_metadata": self.read(folder, "metadata.json"),
                "questions": questions, "solved_events": events}

    def rollout(self, name, index, number):
        folder = self.folder(name)
        base = f"trace/{index:02d}/rollout-{number:02d}"
        if not self.path(folder, base).is_dir():
            raise FileNotFoundError("Rollout is not copied locally")
        response = self.read(folder, f"{base}/response.json")
        request = self.read(folder, f"{base}/request.json")
        telemetry = self.read(folder, f"{base}/telemetry.json")
        available = [f for f in sorted(ROLLOUT_FILES) if self.path(folder, f"{base}/{f}").is_file()]
        return {"rollout": number, "request": request, "response": response,
                "telemetry": telemetry, "files": available}

    def gpu(self, name):
        folder = self.folder(name)
        rows = self.lines(folder, "gpu.jsonl")
        # Keep extrema as well as evenly spaced points for a compact browser plot.
        selected = set(range(len(rows))) if len(rows) <= 600 else {round(i * (len(rows) - 1) / 599) for i in range(600)}
        for field in ("vram_used_mib", "gpu_util_pct"):
            measured = [(i, row[field]) for i, row in enumerate(rows) if row.get(field) is not None]
            if measured:
                selected.update((min(measured, key=lambda v: v[1])[0], max(measured, key=lambda v: v[1])[0]))
        return {"sample_count": len(rows), "samples": [rows[i] for i in sorted(selected)],
                "scope": "Shared GPU device, including concurrent rollouts and preallocated cache."}

    def artifact(self, name, relative):
        allowed = relative in ATTEMPT_FILES or bool(re.fullmatch(
            r"trace/\d{2,}/(?:question\.json|verification\.jsonl|rollout-\d{2,}/(?:request\.json|response\.json|telemetry\.json|tokens\.json|stream\.jsonl))", relative))
        if not allowed:
            raise ValueError("Unsupported artifact")
        path = self.path(self.folder(name), relative)
        if not path.is_file():
            raise FileNotFoundError("Artifact is not copied locally")
        return path
