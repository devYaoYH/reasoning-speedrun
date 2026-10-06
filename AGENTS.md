# Working on reasoning-speedrun

See [CONTRIBUTING.md](CONTRIBUTING.md) for ground rules and [docs/usage.md](docs/usage.md)
for behavior. Essentials for agents:

- Source lives in `src/reasoning_speedrun/`; install with
  `pip install -e '.[grader,data,analysis,dev]'`.
- Check with `python -m pytest` and `node tests/test_results_history.js`. Tests are
  offline; they need neither a GPU nor the licensed AIME files (`tests/helpers.py`
  provides fake provenance).
- v1 is the measured reference policy: never change its behavior to add a feature.
  Add a versioned extension under `extensions/<version>/`, reuse
  `reasoning_speedrun.lib`, and re-pin with
  `python -m reasoning_speedrun.tools.pin --write --reason "..."` after reviewed
  source changes (the manifest tests fail on stale pins).
- GPU experiments are separate from code changes: develop and test locally, commit
  and push, then run on the GPU host from a clean checkout of that commit. Check
  `nvidia-smi` and running services first and never stop an unrelated inference
  server. Use `--reuse-server` only to attach to an idle matching server.
- Keep experiment output (`attempts/`, full SSE streams, grader audits, logs,
  weights, virtualenvs) out of Git; `examples/attempts/` holds the single curated
  sample. Never commit the AIME benchmark files (CC BY-NC-SA 4.0): fetch them with
  `reasoning-speedrun fetch-data`.
- Report unmet targets honestly and do not rank attempts that did not reach the
  target. Measurements need matched datasets, seeds, hardware and timing boundary.
