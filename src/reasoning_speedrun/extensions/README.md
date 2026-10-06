# Versioned policies

Canonical selection is v1 (`reasoning-speedrun`). Select another policy with
`--version VERSION` or run its module directly. Each verifies its own source
manifest (drift is reported, not fatal; see the usage guide) and has its own
defaults, presets and flags (`reasoning-speedrun --version v1.6 --help`).

| Version | Module | Difference from v1 |
| --- | --- | --- |
| [v1.6](v1_6/README.md) | `reasoning_speedrun.extensions.v1_6` | Initial coverage barrier, then a question-count slot pool; up to four fresh samples per question, each continued from 8K to a cumulative 64K/context limit |

## Adding a policy

Put it under `extensions/<version>/` with a public entrypoint, manifest,
presets where needed, and focused offline tests. Reuse `reasoning_speedrun.lib`
(requests, transport, continuations, services, storage, metrics, datasets) and keep
policy decisions in the extension rather than changing v1. Register the selector
in `reasoning_speedrun.lib.entrypoints.EXTENSIONS` and extend
`reasoning_speedrun.tools.pin` to snapshot its manifest. Adding a version does not
promote it: changing `CANONICAL` is an explicit decision backed by matched
validation (same dataset, seeds, hardware and timing boundary).
