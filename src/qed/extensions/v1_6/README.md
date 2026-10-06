# V1.6: four fresh samples, continuations to 64K

This version uses v1.5-style allocation and v1's integer extraction and grading.
Each question has **at most four fresh samples**. Continuation requests do not
consume that allowance. Each fresh trajectory starts with up to 8,192 output
tokens, then sends **one exact-token continuation for the remaining output
budget**, with no further 16K boundaries. It runs until natural completion or its
cumulative 65,536-output-token budget.
The served total context includes the prompt: on the existing 64K-context model,
available output is slightly less than 64K. All generated tokens in the trajectory
count toward its budget. A natural end or exhausted context/budget can free a
slot for another fresh sample; a correct verdict cancels every stream for that
question. The overall attempt stops at 18 distinct correct verdicts by default.

The initial phase launches exactly one 8K request per selected question (30×1
for AIME 2025), with a **coverage barrier**. Freed slots stay idle until all initial
generations and their queued grader checks settle, matching v1’s first-round
barrier. Correct verdicts still cancel their question immediately, and reaching
the solve target ends the attempt without opening the pool.

After that one barrier, the pool opens with a slot count defaulting to the
selected question count (30 for AIME 2025). Ready continuations take priority,
followed by fresh samples on least-active unsolved questions, with rotating ties.
There are no subsequent global barriers: slots refill as requests end or
questions solve. Multiple fresh trajectories for one
question may run concurrently, but their cumulative fresh count never exceeds
four. This is a finite allowance, not a requirement to use all four.

```bash
qed --version v1.6 --benchmark-year 2025 --seed 20261011 --benchmark
# equivalently: python -m qed.extensions.v1_6 ...
```

Use the dedicated 95%-memory
[`vllm-v1_6-long64k.yaml`](../../profiles/models/r0b0tlab/VibeThinker-3B-NVFP4/vllm-v1_6-long64k.yaml)
profile (bundled; used automatically). Its output ceiling is 65,536; total context
stays 65,536. A profile of the same name under `~/models/<organization>/<model>/`
takes precedence. The runner owns warmup, inference/grader services and cleanup, or
can attach to an idle matching server with `--reuse-server`.

Other bundled v1.6 profiles: AWQ + Marlin
([`vllm-v1_6-awq-marlin.yaml`](../../profiles/models/AABoyles/VibeThinker-3B-AWQ/vllm-v1_6-awq-marlin.yaml))
and BF16 + FlashInfer
([`vllm-v1_6-bf16-flashinfer.yaml`](../../profiles/models/WeiboAI/VibeThinker-3B/vllm-v1_6-bf16-flashinfer.yaml));
select them with `--model ORG/NAME --model-profile FILE`.

| Control | Default | Meaning |
| --- | ---: | --- |
| `--max-fresh-samples-per-question` | 4 | Total fresh trajectories, allowed range 1–4 |
| `--max-rollout-tokens` | 65536 | Cumulative output per trajectory, clipped to served total context |
| `--max-concurrent-requests` | Question count | Shared inference slots; must fit initial question coverage |
| `--first-pass-max-tokens` | 8192 | First segment of every fresh trajectory |
| `--model-profile` | `vllm-v1_6-long64k.yaml` | Dedicated profile with a 64K output ceiling |
| `--question-timeout` | 1800s | Overall generation/verification timeout for each question |

`--max-attempts-per-question` is an alias for the fresh-sample limit **only in
this extension**. `--max-tokens` aliases the cumulative `--max-rollout-tokens`
budget; it is not an additional segment size. `--max-rounds`, `--schedule`, `--rollouts`, `--parallelism` and
`--no-continuation` are rejected because they describe different scheduling
contracts. Other service, dataset, sampling, prompt and benchmark flags are
shared with the [usage guide](../../../../docs/usage.md). Custom datasets use the same
gold-free grader question API and retain the integer 0–999 answer restriction.

The improved prompt and shared sampling defaults come from the canonical v1
preset. Separate [policy defaults](policy.json), their hash, actual slot count,
served prompt lengths, resolved configuration and source identity are saved.
Fresh seeds retain `base_seed + question_index * 4 + fresh_sample_id`.
The continuation uses a disjoint seed band whose stride is four times one
plus the maximum selected question index. Changing which sibling finishes first
therefore does not renumber another trajectory's seeds.

Saved `rollout-NN` folders index **HTTP segments**. Their telemetry identifies
`fresh_sample`, `segment`, `continuation_of_rollout`, generated tokens before the

The [manifest](manifest.json) pins this policy, the shared extension lifecycle and
the unchanged canonical primitive dependency. Launch it with `--version v1.6`
or `python -m qed.extensions.v1_6`; the canonical default remains v1.

## Measured behavior

On AIME 2025 with five declared seeds (20261011-20261015) and the NVFP4 model,
v1.6 reached 18 in 5/5 trials: **median 77.498s, range 63.445-101.123s**, with a
smaller tail than the matching v1 batch (worst 128.744s; sample SD 15.0s vs
26.5s). The median is essentially unchanged versus v1, and because scheduling and
the continuation seed bands changed together, the tail reduction cannot be
attributed to the initial barrier alone. Single BF16 and five-seed AWQ comparisons
exist for other deployments; one seed does not establish a quantization speedup.
These numbers come from the original experiment commits; the raw attempt archive
is not part of this repository.

## Difference from other policies

| Policy | Scheduling | Request budget | Four-count interpretation |
| --- | --- | --- | --- |
| v1 | Round barriers, one stream per question | 8K first request, up to 16K further per request | Four HTTP requests including continuations |
| v1.6 | Initial coverage/check barrier, then a question-count pool; continuations first, then least-active fresh samples | Every fresh sample starts at 8K; one long continuation to cumulative 64K/context cap | Four fresh trajectories; the continuation is separate |

Offline policy tests: `python -m pytest tests/test_speedrun_v1_6.py`.
