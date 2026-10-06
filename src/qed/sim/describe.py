"""``qed sim-config``: print the resolved simulation configuration and what it implies.

    qed sim-config                                   # the defaults
    qed sim-config --sim behavior.difficulty=mixed   # expands the preset
    qed sim-config --sim-config my_sim.yaml --sim decode_tps=60
"""

import argparse
from pathlib import Path
import sys

import yaml

from qed.sim.config import SimConfig, expected


def main(argv=None):
    parser = argparse.ArgumentParser(prog="qed sim-config", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim-config", type=Path, help="YAML/JSON file to start from")
    parser.add_argument("--sim", action="append", default=[], metavar="KEY=VALUE", help="Override one knob")
    args = parser.parse_args(argv)
    try:
        config = SimConfig.from_sources(args.sim_config, args.sim)
    except (ValueError, OSError, TypeError) as exc:
        sys.exit(f"Invalid simulation configuration: {exc}")
    summary = expected(config.behavior)
    print("# Resolved configuration (defaults < --sim-config < --sim)")
    print(yaml.safe_dump(config.to_dict(), sort_keys=False).rstrip())
    print("\n# What the model knobs add up to")
    def tidy(value):
        if isinstance(value, float):
            return round(value, 4)
        if isinstance(value, dict):
            return {k: tidy(v) for k, v in value.items()}
        if isinstance(value, list):
            return [tidy(v) for v in value]
        return value

    rounded = tidy(summary)
    print(yaml.safe_dump({"expected": rounded}, sort_keys=False).rstrip())


if __name__ == "__main__":
    main()
