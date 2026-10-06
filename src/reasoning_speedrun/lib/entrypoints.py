"""Explicit policy registry: adding an extension never changes the canonical default."""

from importlib import import_module

CANONICAL = "v1"
EXTENSIONS = {"v1.6": "reasoning_speedrun.extensions.v1_6.cli"}


def launch_extension(version, argv=None):
    """Delegate to the policy's own entrypoint, which pins its source."""
    return import_module(EXTENSIONS[version]).main(argv)
