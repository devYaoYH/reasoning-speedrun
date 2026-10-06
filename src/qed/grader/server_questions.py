"""Versioned grader with gold-free GET /questions; v1 verification toll is unchanged.

GRADER_CONFIG selects the dataset. The solver needs only the service URL; question
statements and their provenance are returned without answers or other row fields.
"""

import hashlib
import json
import os
from pathlib import Path
import sys

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from qed.grader import server as base
from qed.lib.datasets import dataset_provenance


def question_digest(questions):
    raw = json.dumps(
        questions, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def load_dataset(cfg):
    d = cfg["dataset"]
    src, fmt = d["source"], d["format"]
    if fmt != "hf":
        path = Path(src)
        if not path.is_absolute():
            path = Path(base.HERE) / path
        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if fmt == "jsonl":
        rows = [
            json.loads(line) for line in path.read_text().splitlines() if line.strip()
        ]
    elif fmt == "csv":
        import csv

        with path.open() as stream:
            rows = list(csv.DictReader(stream))
    elif fmt == "parquet":
        import pandas as pd

        rows = pd.read_parquet(path).to_dict("records")
    elif fmt == "hf":
        from datasets import load_dataset

        rows = list(
            load_dataset(
                src, revision=d.get("revision"), token=os.environ.get("HF_TOKEN")
            )[d.get("split", "train")]
        )
        source_hash = hashlib.sha256(
            json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
    else:
        raise ValueError(f"Unknown dataset format: {fmt}")
    idx, answer, problem = (
        d.get("idx_field", "problem_idx"),
        d.get("gold_field", "answer"),
        d.get("problem_field", "problem"),
    )
    gold, questions = {}, []
    for row in rows:
        index = int(row[idx])
        if (
            index < 1
            or index in gold
            or not isinstance(row.get(problem), str)
            or not row[problem].strip()
        ):
            raise ValueError(
                "Questions require unique positive indices and nonempty statements"
            )
        if type(row.get(answer)) not in (str, int) or not str(row[answer]).strip():
            raise ValueError("Missing exact answer")
        gold[index] = str(row[answer])
        questions.append({"problem_idx": index, "problem": row[problem]})
    if not gold:
        raise ValueError("No questions loaded")
    questions.sort(key=lambda q: q["problem_idx"])
    if d.get("manifest"):
        evidence = dataset_provenance(None, None, d["manifest"])
        if source_hash != evidence["grader_sha256"]:
            raise ValueError("Grader source disagrees with pinned manifest")
    elif d.get("year") in (2024, 2025, 2026):
        from qed.lib.aime import dataset_provenance as aime_provenance

        evidence = aime_provenance(d["year"])
        if source_hash != evidence["grader_sha256"]:
            raise ValueError("Grader source disagrees with pinned AIME key")
    else:
        evidence = {
            "id": d.get("id", "grader_dataset"),
            "year": d.get("year"),
            "role": None,
            "source": src,
            "revision": d.get("revision"),
            "split": d.get("split", "train"),
            "rows": len(questions),
            "prompt_path": "grader:/questions",
            "prompt_sha256": question_digest(questions),
            "grader_path": src,
            "grader_sha256": source_hash,
            "inferred_from_legacy_runner": False,
        }
    return gold, questions, evidence


class Handler(base.Handler):
    questions = None
    provenance = None

    def do_GET(self):
        if self.path == "/questions":
            self._send(
                200,
                {
                    "questions": self.questions,
                    "dataset": self.provenance,
                    "questions_sha256": question_digest(self.questions),
                },
            )
        else:
            super().do_GET()


def main():
    cfg_path = os.environ.get("GRADER_CONFIG", str(Path(base.HERE) / "config.yaml"))
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    base._load_secrets()
    audit = Path(cfg.get("audit_log", "logs/queries.jsonl"))
    if not audit.is_absolute():
        audit = Path(audit).resolve()  # relative audit paths follow the working directory
    audit.parent.mkdir(parents=True, exist_ok=True)
    gold, questions, evidence = load_dataset(cfg)
    dataset = {
        **cfg["dataset"],
        "id": evidence["id"],
        "year": evidence["year"],
        "sha256": evidence["grader_sha256"],
        "questions_sha256": question_digest(questions),
    }
    Handler.questions, Handler.provenance = questions, evidence
    Handler.oracle = base.Oracle(gold, cfg["cost_c"], str(audit), dataset)
    server = base.ThreadingHTTPServer(
        (cfg.get("host", "127.0.0.1"), int(cfg.get("port", 8077))), Handler
    )
    print(
        f"Grader v2: {len(questions)} questions; GET /questions, POST /verify; toll {cfg['cost_c']}s",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
