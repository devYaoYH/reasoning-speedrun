# Contributing

```bash
pip install -e '.[grader,data,analysis,dev]'
python -m pytest && node tests/test_results_history.js
```

All tests are offline and need no GPU, network or benchmark data.

## Ground rules

- **Do not change v1 behavior to add a feature.** v1 is the measured reference. Add a
  versioned policy under `src/reasoning_speedrun/extensions/<version>/` (see
  `extensions/README.md`) and reuse `reasoning_speedrun.lib`.
- Each policy's source is pinned in a `manifest.json`. After a reviewed change run
  `python -m reasoning_speedrun.tools.pin --write --reason "..."`; the test suite
  fails if the manifests are stale. Drift is only a warning at runtime unless
  `SPEEDRUN_STRICT_INTEGRITY=1`.
- Performance claims need a GPU experiment: matched dataset, seeds, hardware and
  timing boundary, with unmet targets reported rather than ranked. Offline tests
  validate plumbing only.
- Keep gold answers out of solver-visible data and logs.
- Keep experiment output out of Git (`attempts/` is ignored). Only curated examples
  belong under `examples/`; check sizes before adding one.
- The AIME data is licensed CC BY-NC-SA 4.0 and must not be committed.

## Layout

See [docs/usage.md](docs/usage.md#package-layout).
