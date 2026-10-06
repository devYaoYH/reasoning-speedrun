"""Package resources, the user's working directory, and durable JSON writes."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]


def workdir() -> Path:
    """Where attempts, locks and relative user paths live (``$QED_HOME`` or cwd)."""
    return Path(os.environ.get("QED_HOME") or Path.cwd()).resolve()


def shown(path) -> str:
    """A path as short as is unambiguous: relative to the working directory when inside it."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)
