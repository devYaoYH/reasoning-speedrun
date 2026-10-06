"""Public CLI; presets override defaults, explicit flags override presets."""

import argparse
import asyncio
import hashlib
from importlib import import_module, metadata
import json
from pathlib import Path
import signal
import sys

from reasoning_speedrun.config import parse_args as parse_config
from reasoning_speedrun.integrity import verify_core
from reasoning_speedrun.lib.common import PACKAGE
from reasoning_speedrun.lib.entrypoints import CANONICAL, EXTENSIONS, launch_extension
from reasoning_speedrun.lib.services import attempt_lock
from reasoning_speedrun.run import RUNNER_ID, run

DEFAULT = Path(__file__).with_name("presets") / "prompt_adherence.json"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--preset", type=Path, default=DEFAULT)
    parser.add_argument("--system-prompt-file", type=Path)
    options, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    path = options.preset.expanduser().resolve()
    raw = path.read_bytes()
    preset = json.loads(raw)
    if (
        set(preset) != {"schema_version", "core_id", "prompt_file", "argv"}
        or preset["schema_version"] != 1
        or preset["core_id"] != RUNNER_ID
    ):
        raise ValueError("Preset must select canonical v1 and use schema version 1")
    if not isinstance(preset["argv"], list) or not all(
        isinstance(x, str) for x in preset["argv"]
    ):
        raise ValueError("Preset argv must be a list of argument strings")
    prompt_path = (
        options.system_prompt_file.expanduser().resolve()
        if options.system_prompt_file
        else (path.parent / preset["prompt_file"]).resolve()
    )
    prompt_raw = prompt_path.read_bytes()
    prompt = prompt_raw.decode("utf-8")
    if not prompt.strip():
        raise ValueError("System prompt cannot be empty")
    args = parse_config(preset["argv"] + remaining)
    args.system_prompt = prompt
    args.system_prompt_file, args.preset_file = str(prompt_path), str(path)
    args.system_prompt_sha256 = hashlib.sha256(prompt_raw).hexdigest()
    args.preset_sha256 = hashlib.sha256(raw).hexdigest()
    args.runtime_package_versions = {}
    for package in ("vllm", "torch", "httpx", "PyYAML", "nvidia-ml-py", "jsonschema"):
        try:
            args.runtime_package_versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            args.runtime_package_versions[package] = None
    return args


async def execute(args):
    asyncio.get_running_loop().add_signal_handler(
        signal.SIGTERM, asyncio.current_task().cancel
    )
    await run(args)


SUBCOMMANDS = {
    "view": ("reasoning_speedrun.viewer.server", "Browse saved attempts in a local viewer"),
    "fetch-data": ("reasoning_speedrun.fetch_data", "Download the pinned benchmark files"),
}


def main(argv=None):
    values = sys.argv[1:] if argv is None else argv
    if values and values[0] in SUBCOMMANDS:
        return import_module(SUBCOMMANDS[values[0]][0]).main(values[1:])
    selection = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    selection.add_argument("--version", choices=("v1", *EXTENSIONS), default=CANONICAL)
    selected, remaining = selection.parse_known_args(values)
    if selected.version != "v1":
        return launch_extension(selected.version, remaining)
    if "--help" in remaining or "-h" in remaining:
        print(
            "Subcommands: view (attempt viewer), fetch-data (benchmark files); "
            "see `reasoning-speedrun view --help`\n"
            "Policy selection: --version {v1,v1.6,naive}\n"
            "Configuration: --preset FILE --system-prompt-file FILE\n"
            "Explicit flags override preset defaults. Help launches no services."
        )
    verify_core(PACKAGE)
    args = parse_args(remaining)
    with attempt_lock():
        asyncio.run(execute(args))
