# qed

*Quod erat demonstrandum:* stop the moment it's proved.

Scheduling policies for streams of reasoning-model requests: **bounded
parallelism, streaming chunks, and early exit via optimistic intermediate answer
extraction**, measured on a real GPU.

A reasoning model thinks for thousands of tokens, but its answer often appears
long before it stops. Here a policy decides how many streams run at once, reads
candidate answers out of the stream while the model is still reasoning, sends
each one to a verifier that is slow and shared (a toll-gated grader: 3 s per
check, one global queue), and cancels everything for a question the moment a
check passes. The grader's toll is the cost of being wrong or early: optimism
only pays if the checks it triggers are cheap enough relative to the generation
they save, and concurrency only pays until the GPU is contended.

| Piece | What it answers |
| --- | --- |
| Real runs (`qed`) | How long does a policy actually take to reach N verified answers on your GPU? Per-request timestamps, token IDs, verdict times, GPU/engine telemetry |
| Policies (`--version`) | v1 (round barriers), v1.6 (coverage barrier + slot pool), **naive** (complete fan-out, final answers only) |
| Simulated backend (`--simulate`) | How does a policy behave under serving characteristics you choose (decode and prefill speed, sequence cap, prefix cache, model accuracy and length), with no GPU? |
| Viewer (`qed view`) | Trajectories, verdict timelines, GPU samples, matched-control comparisons |

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

qed isn't on PyPI (the `qed` name there belongs to something else), so clone the repo.
Python 3.11+ is all you need to browse results and to **simulate** runs; only real runs
need an NVIDIA GPU, Linux and [vLLM](https://docs.vllm.ai) in the same environment (qed
launches `vllm serve` itself).

```bash
git clone https://github.com/devYaoYH/reasoning-speedrun && cd reasoning-speedrun
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[grader,data,analysis]'   # runner + grader deps, dataset download, plots
```

## Quickstart

**1. Look around (no GPU, no data).** The repo ships one recorded real run (18 correct
in 62 s on an A100) so you can see what qed produces:

```bash
qed view --attempts examples/attempts      # then open http://127.0.0.1:8765
```

*Attempts* shows one run's trajectories, grader verdicts and request timeline;
*Overall results* compares runs by time to target.

**2. Run a policy in simulation (no GPU, no data).** `--simulate` replaces vLLM and the
GPU with an in-process mock; the scheduler, extraction, grader and traces are real. It
takes real wall-clock time (about two minutes here) and runs on 30 synthetic questions
that ship with the repo:

```bash
qed --simulate --grader-config src/qed/examples/synthetic_grader.yaml \
    --system-prompt-file src/qed/examples/integer_prompt.txt          # canonical v1
qed view                                   # your run is now in ./attempts
```

Compare policies by changing `--version` (`v1`, `v1.6`, `naive`) and the serving model
by changing knobs, e.g. `--sim decode_tps=60 --sim max_num_seqs=32`. `qed sim-config`
shows every knob and what it implies. Simulated runs are labelled and never mixed with
hardware results; see [Simulation](docs/usage.md#simulation) for the knobs and caveats
(notably: decode speed does not slow down under load).

**3. Run on a real GPU.**

```bash
qed fetch-data --year 2025     # AIME is CC BY-NC-SA, so it is fetched, not bundled
# Put the model's weights under ~/models/<org>/<name>/ (see docs/usage.md);
# launch profiles for the reference models ship with qed.
nvidia-smi                     # qed refuses an occupied GPU
qed --seed 20261011            # canonical v1: 30 questions, 8K first request, target 18
qed view
```

qed warms the engine, launches vLLM and a fresh grader, starts the clock and writes
`attempts/<timestamp>/` (`summary.json` has `target_reached` and `time_to_target_s`).

**4. Your own questions.** Give the grader a JSONL file of `problem_idx`, `problem`,
`answer` and point a grader YAML at it; the solver only ever sees indices and statements.
Add `--simulate` to run this without a GPU.

```bash
qed --grader-config src/qed/examples/integer_grader.yaml \
    --system-prompt-file src/qed/examples/integer_prompt.txt \
    --parallelism 2 --target-correct 2
```

**Answer extraction is integer-only (0-999) in the shipped policies.** Other answer
formats need a new policy extension; see the [usage guide](docs/usage.md#datasets).
Something not working? Errors print one line; `QED_DEBUG=1` shows the traceback.

## Policies

| Policy | Parallelism | Answer extraction |
| --- | --- | --- |
| `v1` (default) | 30 streams (one per question), round barriers, exact-ID continuations | Early: closed boxes, answer lines and clauses in reasoning and content |
| `v1.6` | Coverage barrier, then a question-count slot pool; four fresh samples per question | Early, as v1 |
| `naive` | Complete fan-out: every sample of every question at once | Final only: last box of a naturally ended response |

Select with `--version`. New policies plug into one shared pipeline; the
[policy contract](src/qed/extensions/README.md#the-policy-contract)
lists what to supply. No GPU measurement of `naive` is recorded here yet.

## What is in the box

| Piece | Where | What it gives you |
| --- | --- | --- |
| Runner | `src/qed/` | Streaming candidate extraction, bounded round scheduling, exact-token-ID continuations with prefix-cache measurement, first-solved timestamps linked to grader queries, benchmark mode that keeps evidence in RAM until timing ends |
| Policies | `extensions/` | `v1.6` and `naive`; the template for new ones |
| Grader | `src/qed/grader/` | Standalone toll-gated, FIFO, gold-free answer oracle (MathArena parser), usable by any solver |
| Viewer | `qed view` | Per-attempt trajectories, verdicts and GPU samples; aggregate time-to-target with matched-control comparison |
| Analysis | `python -m qed.analysis.plot_attempt ATTEMPT_DIR OUT_DIR` | Requests in flight, verified-correct curve and request budgets, plotted from a saved attempt |
| Samples | `examples/attempts/`, `src/qed/examples/` | A recorded real run for the viewer; bundled inputs (synthetic questions, grader configs, a simulation config) |
| Data tools | `fetch-data`, `lib.datasets` | Hash-verified, revision-pinned benchmark download; custom-dataset adapter |

More: [usage guide](docs/usage.md) (flags, policy and timing semantics, profiling),
[v1.6 policy](src/qed/extensions/v1_6/README.md),
[grader](src/qed/grader/README.md),
[benchmark provenance](src/qed/data/README.md),
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
