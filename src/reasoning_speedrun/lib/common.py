"""Package resources, the user's working directory, and durable JSON writes."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]


def workdir() -> Path:
    """Where attempts, locks and relative user paths live (``$SPEEDRUN_HOME`` or cwd)."""
    return Path(os.environ.get("SPEEDRUN_HOME") or Path.cwd()).resolve()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)
