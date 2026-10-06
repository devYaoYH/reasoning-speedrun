# reasoning-speedrun

A harness for measuring **how fast a locally served reasoning model can solve N
math problems, as confirmed by a rate-limited grader.** It drives a vLLM server
with many concurrent streams, extracts candidate answers while the model is still
thinking, checks them against a toll-gated grader (3 s per check by default, one
global queue), continues capped generations from their exact token IDs, and
records everything needed to audit the clock: per-request timestamps, token IDs,
verdict times and optional GPU/engine telemetry. A browser viewer and offline
plots work over the saved attempts.

The reference task is **time to 18 distinct grader-confirmed answers on AIME 2025
with a 3B model on one A100 80GB** (floor: 18 × 3 s = 54 s of serial grading).

| Evaluation (reference NVFP4 3B model) | Time to 18 |
| --- | --- |
| AIME 2025, five declared seeds, one server | 5/5 reached 18; **median 77.3 s**, range 62.8-82.5 s |
| Same policy repackaged in this repo, five seeds | 5/5 reached 18; median 73.6 s, range 62.1-128.7 s |
| v1.6 policy (coverage barrier + slot pool), five seeds | 5/5 reached 18; median 77.5 s, range 63.4-101.1 s |
| AIME 2026, unchanged v1 policy, one run | 88.7 s, one wrong check |

These come from the original GPU experiments (the raw attempt archive is not
included here); [`docs/report/`](docs/report/) holds the original write-up.
Offline tests establish the plumbing, not a new GPU timing: reproduce on your
hardware before quoting numbers.

## Install

Linux, Python 3.11+, an NVIDIA GPU and [vLLM](https://docs.vllm.ai) in the same
environment as this package (it launches `vllm serve` itself):

```bash
pip install 'reasoning-speedrun[grader]'      # runner + bundled grader deps
pip install 'reasoning-speedrun[data]'        # only to fetch the AIME benchmark files
pip install 'reasoning-speedrun[analysis]'    # only for offline plots
```

From a checkout use `pip install -e '.[grader,data,analysis,dev]'`.

## Quickstart

```bash
# 1. Benchmark data is licensed upstream (CC BY-NC-SA 4.0), so it is fetched, not bundled.
python -m reasoning_speedrun.fetch_data --year 2025

# 2. Weights: put the model under ~/models/<org>/<name>/ (see docs/usage.md). Launch
#    profiles for the reference models ship with the package.
nvidia-smi                                    # the runner refuses an occupied GPU
reasoning-speedrun --seed 20261011            # canonical v1: 30x1, 8K first request, target 18

# 3. Look at the result.
speedrun-viewer                               # http://127.0.0.1:8765, reads ./attempts
```

The runner warms the engine, launches vLLM and a fresh grader, starts the clock,
and writes `attempts/<timestamp>/` (`summary.json` has `target_reached` and
`time_to_target_s`). Try the viewer first without a GPU on the bundled example:

```bash
speedrun-viewer --attempts examples/attempts
```

### Your own questions

Give the grader a JSONL file of `problem_idx`, `problem`, `answer` and point a
grader YAML at it; the solver only ever sees indices and statements.

```bash
reasoning-speedrun \
  --grader-config src/reasoning_speedrun/examples/integer_grader.yaml \
  --system-prompt-file src/reasoning_speedrun/examples/integer_prompt.txt \
  --parallelism 2 --target-correct 2
```

**Answer extraction is integer-only (0-999) in the shipped policies.** Other
answer formats need a new policy extension; see the
[usage guide](docs/usage.md#datasets).

## What is in the box

| Piece | Where | What it gives you |
| --- | --- | --- |
| Runner | `reasoning_speedrun/` | Bounded round scheduling, streaming candidate extraction, exact-token-ID continuations with prefix-cache measurement, first-solved timestamps linked to grader queries, benchmark mode that keeps evidence in RAM until timing ends |
| Policies | `extensions/v1_6/` | Versioned alternatives selected with `--version`; template for new ones |
| Grader | `reasoning_speedrun/grader/` | Standalone toll-gated, FIFO, gold-free answer oracle (MathArena parser), usable by any solver |
| Viewer | `speedrun-viewer` | Per-attempt trajectories, verdicts and GPU samples; aggregate time-to-target with matched-control comparison |
| Analysis | `reasoning_speedrun.analysis` | Offline plots from saved telemetry |
| Data tools | `fetch_data`, `lib.datasets` | Hash-verified, revision-pinned benchmark download; custom-dataset adapter |

More: [usage guide](docs/usage.md) (flags, policy and timing semantics, profiling),
[v1.6 policy](src/reasoning_speedrun/extensions/v1_6/README.md),
[grader](src/reasoning_speedrun/grader/README.md),
[benchmark provenance](src/reasoning_speedrun/data/README.md),
[contributing](CONTRIBUTING.md).

## Development

```bash
pip install -e '.[grader,data,analysis,dev]'
python -m pytest            # offline; no GPU or network
node tests/test_results_history.js
```

## License

Code: [MIT](LICENSE). The AIME benchmark files are **not** covered by it: they are
fetched from MathArena (CC BY-NC-SA 4.0; problems credit the MAA). The bundled
example attempt includes the 30 AIME 2025 problem statements it was run on, under
those upstream terms. Model weights carry their own licenses.
