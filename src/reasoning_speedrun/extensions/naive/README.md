# Naive baseline: complete fan-out, final answers only

The reference point for what scheduling and early exit buy.

- **Parallelism:** every sample of every question starts at once (30 questions x 4
  samples = 120 streams for AIME 2025). Nothing in the client bounds concurrency;
  the inference server's own scheduler (`max-num-seqs` in the launch profile)
  queues the excess.
- **Extraction:** nothing is read while a sample streams. Only after a sample ends
  *naturally* (`stop`, not a length cap) is the **last closed integer box of its
  final response** proposed as a candidate. Boxes and "Answer:" lines in the
  reasoning are ignored; a capped sample yields nothing.
- **No continuations, rounds or barriers.** Each sample is one request with the
  full output cap (default 16,384).
- **Unchanged around that:** candidates are deduplicated per question, checked
  against the same shared toll-gated grader, a verified-correct answer cancels the
  question's other samples, and the attempt stops at `--target-correct` questions.

```bash
reasoning-speedrun --version naive --seed 20261011
reasoning-speedrun --version naive --samples-per-question 2 --max-tokens 8192
```

Flags that describe other scheduling contracts (`--parallelism`, `--rollouts`,
`--schedule`, `--max-rounds`, `--no-continuation`, `--max-attempts-per-question`,
`--first-pass-max-tokens`) are rejected. All other service, dataset, sampling and
benchmark flags are shared with canonical usage.

Offline tests: `python -m pytest tests/test_naive.py`. No GPU measurement of this
policy has been recorded in this repository; compare it against v1 or v1.6 with
matched seeds and the same model profile (its generation ceiling must cover
`--max-tokens`).
