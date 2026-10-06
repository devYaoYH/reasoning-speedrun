# Benchmark provenance

Only provenance manifests live here; the benchmark files themselves are **not
bundled** because MathArena distributes them under **CC BY-NC-SA 4.0**. Fetch
them into `$SPEEDRUN_DATA` (default `~/.cache/reasoning_speedrun`) with:

```bash
pip install 'reasoning-speedrun[data]'
reasoning-speedrun fetch-data --year 2024 --year 2025 --year 2026
```

The downloader resolves each source at its pinned Hugging Face revision and
refuses to write files whose hashes differ from the manifest.

- `source_2024.json`: [MathArena AIME I](https://huggingface.co/datasets/MathArena/aime_2024_I) + [II](https://huggingface.co/datasets/MathArena/aime_2024_II), used as the ungraded cheap warmup workload.
- `source.json`: [MathArena/aime_2025](https://huggingface.co/datasets/MathArena/aime_2025), the development benchmark.
- `source_2026.json`: [MathArena/aime_2026](https://huggingface.co/datasets/MathArena/aime_2026), a held-out generalization benchmark.

Each year has 30 ordered problems: indices 1-15 are AIME I, 16-30 are AIME II.
Solver loaders return only indices and problem text; answers go to the grader.
MathArena documents [different US/international AIME II variants](https://huggingface.co/datasets/MathArena/aime_2026/discussions/2);
preserve statements and answers together when comparing against external results.
Original contest problems are credited to the Mathematical Association of America.
Downloading a year establishes a local test set; it does not establish that a model
has never seen the problems during pretraining.

For your own questions, use a JSONL file with `problem_idx`, `problem` and `answer`
(see `examples/` and the top-level README).
