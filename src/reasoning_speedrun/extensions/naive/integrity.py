"""Pin the naive policy and verify its unchanged canonical dependency."""

import hashlib
import json
from reasoning_speedrun.integrity import drift, report, verify_core as verify_base


def verify_core(root):
    raw = (root / "extensions/naive/manifest.json").read_bytes()
    manifest = json.loads(raw)
    if manifest["core_id"] != "runner_core_naive" or manifest["schema_version"] != 1:
        raise RuntimeError("Unsupported naive-policy manifest")
    if verify_base(root) != manifest["canonical_manifest_sha256"]:
        report("Canonical dependency manifest", ["manifest.json"])
    report("Naive policy", drift(root, manifest))
    return hashlib.sha256(raw).hexdigest()
