"""Record reviewed source snapshots for the canonical core and each policy extension.

Run ``python -m qed.tools.pin --reason "..." --write`` after a
reviewed behavior change. Without ``--write`` the manifests are printed.
"""

import argparse
import hashlib
import json

from qed.lib.common import PACKAGE, atomic_json

SKIP = {"__pycache__", ".venv", "venv", ".cache"}
EXTENSIONS = ("v1_6", "naive")


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


def extension_files(name):
    files = [
        p
        for p in (PACKAGE / "extensions" / name).rglob("*")
        if p.suffix in (".py", ".json")
        and p.name != "manifest.json"
        and not SKIP.intersection(p.parts)
    ]
    if name == "v1_6":
        files += [
            PACKAGE / "lib/policy_runtime.py",
            PACKAGE / "profiles/models/r0b0tlab/VibeThinker-3B-NVFP4/vllm-v1_6-long64k.yaml",
        ]
    else:
        files.append(PACKAGE / "lib/policy_runtime.py")
    return files


def build_extension(name, reason, core_manifest_bytes):
    path = PACKAGE / "extensions" / name / "manifest.json"
    previous = json.loads(path.read_text())
    return {
        **{k: v for k, v in previous.items() if k not in ("sha256", "change_reason")},
        "canonical_manifest_sha256": hashlib.sha256(core_manifest_bytes).hexdigest(),
        "change_reason": reason,
        "sha256": hashes(extension_files(name)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reason", required=True, help="Describe the reviewed change")
    parser.add_argument("--write", action="store_true", help="Write all manifests")
    args = parser.parse_args()
    if not args.reason.strip():
        parser.error("A nonempty change reason is required")
    core = build_core(args.reason)
    core_bytes = (json.dumps(core, ensure_ascii=False, indent=2) + "\n").encode()
    built = {name: build_extension(name, args.reason, core_bytes) for name in EXTENSIONS}
    if args.write:
        atomic_json(PACKAGE / "manifest.json", core)
        for name, value in built.items():
            atomic_json(PACKAGE / "extensions" / name / "manifest.json", value)
    else:
        print(json.dumps({"core": core, **built}, indent=2))


if __name__ == "__main__":
    main()
