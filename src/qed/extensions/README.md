# Versioned policies

Canonical selection is v1 (`qed`). Select another policy with
`--version VERSION` or run its module directly. Each verifies its own source
manifest (drift is reported, not fatal; see the usage guide) and has its own
defaults, presets and flags (`qed --version v1.6 --help`).

| Version | Module | Parallelism | Answer extraction |
| --- | --- | --- | --- |
| v1 (canonical) | `qed` | One stream per question, 30 concurrent; round barriers; continuations up to four requests | Streaming (early): closed boxes, answer lines, answer clauses, in reasoning and content |
| [v1.6](v1_6/README.md) | `qed.extensions.v1_6` | Coverage barrier, then a question-count slot pool; four fresh samples per question, each continued to a cumulative 64K | Streaming (early), as v1 |
| [naive](naive/README.md) | `qed.extensions.naive` | **Complete fan-out**: every sample of every question at once, bounded only by the server | **Final only**: last integer box of a naturally ended response |

The naive policy is the baseline the others are compared against: it shows what
bounded parallelism and early exit buy.

## The policy contract

A policy is the scheduling and extraction decisions around one shared pipeline
(`question.py`: propose, deduplicate, verify, cancel, persist; `lib/generation.py`:
one stream with exact-token evidence). An extension supplies:

1. **`policy(problems, args, client, output, sampler, official_start, profiler) -> rows`**:
   an async function that decides which streams run, when, and with what budget
   (`scheduler.py` for v1, `v1_6/policy.py` + `allocation.py` for a slot pool,
   `naive/policy.py` for full fan-out). It must stop when `target_correct`
   questions are solved and must not read gold answers.
2. **Extraction**, by subclassing `lib.generation.RolloutGenerator` (override
   `detect`) and setting `QuestionRun.generator`: see `naive/policy.py`.
3. **`parse_args`, `prepare_problems`, `metadata`** and an `integrity.verify_core`
   with a `manifest.json`, wired together by `lib.policy_runtime.run_policy`
   (service lifecycle, the official clock, summaries): see `naive/cli.py`, the
   smallest complete example.
4. A selector in `lib.entrypoints.EXTENSIONS`, a name in `tools/pin.py`, and offline
   tests with the fake-SSE helpers in `tests/helpers.py`.

Keep policy decisions in the extension rather than changing v1. Adding a version
does not promote it: changing `CANONICAL` is an explicit decision backed by matched
validation (same dataset, seeds, hardware and timing boundary).
