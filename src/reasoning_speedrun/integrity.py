"""Pin package source: record a source snapshot and flag drift from the manifest.

Drift is a warning by default (edited installs and forks remain usable) and is
recorded in each attempt's config. Set ``SPEEDRUN_STRICT_INTEGRITY=1`` to make
drift fatal when reproducing a pinned measurement.
"""

import hashlib
import json
import os
import sys
from pathlib import Path

from reasoning_speedrun.lib.common import PACKAGE

MANIFEST = Path("manifest.json")


def strict():
    return os.environ.get("SPEEDRUN_STRICT_INTEGRITY") == "1"


def drift(root, manifest):
    return [
        name
        for name, digest in manifest["sha256"].items()
        if not (root / name).is_file()
        or hashlib.sha256((root / name).read_bytes()).hexdigest() != digest
    ]


def report(label, changed):
    if not changed:
        return
    message = f"{label} source drift: " + ", ".join(changed)
    if strict():
        raise RuntimeError(message)
    print("warning: " + message, file=sys.stderr, flush=True)


def verify_core(root=PACKAGE):
    raw = (root / MANIFEST).read_bytes()
    manifest = json.loads(raw)
    if (
        manifest.get("core_id") != "runner_core_v1"
        or manifest.get("schema_version") != 1
    ):
        raise RuntimeError("Unsupported canonical runner manifest")
    report("Canonical runner", drift(root, manifest))
    return hashlib.sha256(raw).hexdigest()
