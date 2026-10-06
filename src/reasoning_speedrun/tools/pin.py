"""Record reviewed source snapshots for the canonical core and the v1.6 policy.

Run ``python -m reasoning_speedrun.tools.pin --reason "..." --write`` after a
reviewed behavior change. Without ``--write`` the manifests are printed.
"""

import argparse
import hashlib
import json

from reasoning_speedrun.lib.common import PACKAGE, atomic_json

SKIP = {"__pycache__", ".venv", "venv", ".cache"}
V1_6_MANIFEST = PACKAGE / "extensions/v1_6/manifest.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def hashes(files):
    return {str(p.relative_to(PACKAGE)): digest(p) for p in sorted(files)}


def build_core(reason):
    files = [
        p
        for p in PACKAGE.rglob("*.py")
        if not {"extensions", "viewer", "analysis", "tools"}.intersection(
            p.relative_to(PACKAGE).parts
        )
        and not SKIP.intersection(p.parts)
    ]
    files.append(PACKAGE / "metadata.schema.json")
    return {
        "schema_version": 1,
        "core_id": "runner_core_v1",
        "policy": "V1 bounded round scheduling, integer candidate extraction, exact-ID continuations and four-request ceiling. Includes a custom integer-dataset adapter.",
        "change_reason": reason,
        "sha256": hashes(files),
    }


def build_v1_6(reason, core_manifest_bytes):
    previous = json.loads(V1_6_MANIFEST.read_text())
    files = [
        p
        for p in (PACKAGE / "extensions/v1_6").rglob("*")
        if p.suffix in (".py", ".json")
        and p.name != "manifest.json"
        and not SKIP.intersection(p.parts)
    ]
    files += [
        PACKAGE / "lib/policy_runtime.py",
        PACKAGE / "profiles/models/r0b0tlab/VibeThinker-3B-NVFP4/vllm-v1_6-long64k.yaml",
    ]
    return {
        **{k: v for k, v in previous.items() if k not in ("sha256", "change_reason")},
        "canonical_manifest_sha256": hashlib.sha256(core_manifest_bytes).hexdigest(),
        "change_reason": reason,
        "sha256": hashes(files),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reason", required=True, help="Describe the reviewed change")
    parser.add_argument("--write", action="store_true", help="Write both manifests")
    args = parser.parse_args()
    if not args.reason.strip():
        parser.error("A nonempty change reason is required")
    core = build_core(args.reason)
    core_bytes = (json.dumps(core, ensure_ascii=False, indent=2) + "\n").encode()
    v16 = build_v1_6(args.reason, core_bytes)
    if args.write:
        atomic_json(PACKAGE / "manifest.json", core)
        atomic_json(V1_6_MANIFEST, v16)
    else:
        print(json.dumps({"core": core, "v1_6": v16}, indent=2))


if __name__ == "__main__":
    main()
