"""Pin the v1.6 policy and verify its canonical primitive dependency."""

import hashlib
import json
from qed.integrity import drift, report, verify_core as verify_base


def verify_core(root):
    raw = (root / "extensions/v1_6/manifest.json").read_bytes()
    manifest = json.loads(raw)
    if manifest["core_id"] != "runner_core_v1_6" or manifest["schema_version"] != 1:
        raise RuntimeError("Unsupported v1.6 manifest")
    if verify_base(root) != manifest["canonical_manifest_sha256"]:
        report("Canonical dependency manifest", ["manifest.json"])
    report("V1.6", drift(root, manifest))
    return hashlib.sha256(raw).hexdigest()
