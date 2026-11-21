# Calibrating the per-iteration time estimate

The engine's scheduler needs an estimate of the time cost of adding prompt tokens to a mixed iteration. This directory fits and evaluates that estimate without importing vLLM or requiring a GPU. Use Python 3.10+ from the repository root.

## What one observation means

Every eligible measured iteration contributes one JSON object on its own line. The following record uses **invented** values.

```json
{"clock":"scheduler_result_wall","run_id":"SYNTHETIC-F","decode_count":2,"context_tokens":512,"prefill_tokens":64,"step_ms":11.0}
```

| Field | Meaning |
| --- | --- |
| `clock` | Exactly `scheduler_result_wall`. The scheduler-to-result wall interval contains GPU work, synchronization and CPU/runtime overhead within that boundary. It does not include time a request waited before selection and must not be reported as CUDA-kernel time. |
| `run_id` | Unique identifier for an entire server/workload run. Give each independent replay a fresh ID. |
| `decode_count` | At the start of scheduling, the number of running requests having completed prompt computation and eligible for their next decode computation. Measured before choosing a new prefill budget. |
| `context_tokens` | Sum of the already computed context lengths for those decode requests at that same moment. This is a scheduler feature, not allocated KV memory. |
| `prefill_tokens` | Total actual prompt tokens selected across all requests in the iteration. This can be lower than the permitted budget. |
| `step_ms` | Positive finite elapsed milliseconds measured using the chosen boundary. |

Two batches with identical feature bins can cost different amounts. Variations include the number of prefill sequences, prompt prefix lengths, GPU graph paths and runtime jitter. Treat the estimate as a coarse service-time model and judge utility by a separate end-to-end comparison.

The adapter skips empty steps, steps with a preemption, multi-token decode recomputation or a prompt/decode replay, and steps whose scheduled decode count or summed contexts differ from the pre-scheduling observation. Emitted rows carry cumulative exclusion counts by reason in `excluded_cumulative`. The counters on the last row may omit trailing excluded steps. Report these counts alongside coverage, especially under memory pressure. Held-out coverage uses the emitted rows as its denominator. End-to-end client TTFT, inter-token latency and completion rate must still account for the entire offered workload according to the benchmark's metric definitions.

## Collect, fit, validate

1. Pin the model revision, GPU, vLLM commit, precision, KV/cache settings and scheduling options. Keep warmup, sampling/instrumentation, GPU contention and request mix documented. Use the engine instructions to collect `steps.jsonl` from **instrumented native/fixed-budget runs**. Run several caps and rates that exercise the intended decode and prefill ranges. Preserve the original trace and each workload seed. The core accepts step records directly and cannot infer step duration or true selected prefill tokens from client TTFT/ITL.
2. Divide complete independent runs into calibration, model-validation and final benchmark sets. Never use a final benchmark run to choose table edges, target milliseconds, fallback size, or SLOs. When comparing measured CPU/GPU environments, use a new profile. The CLI requires disjoint run IDs and rejects identical trace files across fit/evaluate, but hardware/workload comparability remains the operator's responsibility.
3. Save an environment JSON with the actual `gpu`, `model_revision`, `vllm_revision`, `dtype`, `capture_mode` and workload descriptors. It is copied into `metadata`. Check it before applying a profile. Use the fit and evaluate commands below. The paths denote the operator's real run directories.

```bash
python -m calibration fit \
  --traces runs/cal-low/steps.jsonl runs/cal-mid/steps.jsonl runs/cal-high/steps.jsonl \
  --metadata runs/environment.json --skip-first 20 --min-samples 20 \
  --out runs/profile.json

python -m calibration evaluate --profile runs/profile.json \
  --traces runs/validation-a/steps.jsonl runs/validation-b/steps.jsonl \
  --skip-first 20 --target-ms 25 --out runs/validation.json
```

`--skip-first` excludes that many initial measured rows per run, including initial compute work when appropriate. Default is 0. Specify the number from your actual warmup protocol. A 25 ms step target in these commands is illustrative; it is a controller guardrail, not a client TTFT/TPOT promise. To require a validation coverage threshold, add `--min-coverage 0.8` or the threshold appropriate for the experiment. Reports are written even if validation coverage is below the requirement. Zero covered rows always produce a nonzero evaluation exit status.

## Fit and policy lookup

`decode_bins`, `context_bins` and `prefill_bins` contain inclusive upper bounds. Each begins at zero. For edges `[0,32,64]`, a feature of 32 belongs in the 32 box and 33 in the 64 box. Above the largest edge is unsupported. Supply comma-separated `--decode-bins`, `--context-bins` and `--prefill-bins` to change defaults. Fit edges on calibration data only. Include zero explicitly to separate steps without decode or prefill. Choose candidate chunk sizes consistent with the bins so the calibration file has useful support.

For each box, fit takes nearest-rank empirical p95 of measured per-step times. Within each decode/context box it takes a prefix maximum across supported prefill sizes. Sampling noise cannot make a larger supported budget appear cheaper. Omitted bins remain unsupported. Each cell contains `n`, `raw_p95_ms`, conservative `p95_ms` and the smallest/largest observed prefill amount in that box. A cell with fewer than `min_samples` is omitted. Twenty observations give only a coarse estimate of p95. For reporting, seek substantially more repeated steps and independent replays.

```python
import json
from calibration.table import Table

with open("runs/profile.json", encoding="utf-8") as source:
    table = Table(json.load(source))
estimated_ms = table.lookup(decode_count=8, context_tokens=16384, prefill_tokens=256)
# None means the state/size has no supported estimate.
```

A lookup can return a bin estimate even if this exact shape was not sampled. Check the observed min/max for each prefill bucket before increasing the deployed cap. Missing cells supply no cost estimate. If the current decode shape has no supported candidate, the scheduler uses its configured fallback budget. The actual batch can be smaller than the selected cap. Compare recorded decisions, token usage and fallback incidence with client-observed latency. The periodic progress override releases a shared allowance. Individual FCFS requests can still wait behind other work.

## Evaluation and failure analysis

| Field | Interpretation |
| --- | --- |
| `coverage` and `groups` | Fraction of all held-out observations that have a usable estimate. Mixed steps are reported separately. High overall coverage can mask rarely covered mixed batches. |
| `mean_prediction_minus_actual_ms` | Signed residual averaged over covered rows. A predominantly positive residual is expected for a conservative p95 envelope. |
| `mean_absolute_error_ms`, `p95_absolute_error_ms` | Absolute differences between the upper estimate and individual observed step times on covered rows. These are not errors of a median predictor. |
| `fraction_above_estimate_on_covered` | Fraction of measured steps above their fitted upper estimate. A per-bin empirical p95 on calibration data does not guarantee 95% future coverage. |
| `target_check.false_safe_fraction` | Among supported observations whose predicted cost met the specified target, fraction that actually exceeded it. Null means no supported prediction passed this target. |

Typical diagnostic: low held-out mixed coverage means a policy will often trigger fallback. Inspect observed bins, add fixed-budget calibration runs or coarsen edges, refit, then validate against an untouched replay. High coverage but too many false-safe decisions suggests different request geometry or workload drift; inspect errors by length, checkpoint, runtime configuration and timing boundary. Making the envelope more conservative may reduce decoder stalls and lengthen prompt waiting. Track both effects in the serving benchmark. Stateful feedback changes which shapes appear, so always repeat coverage and goodput assessment on policy traces after freezing the model. Avoid treating serial steps from one replay as statistically independent confidence replicates.

### Synthetic smoke test

The fixture filenames, run IDs and metadata all label these input times synthetic. No project performance result follows from this test.

```bash
python -m calibration fit --traces calibration/fixtures/SYNTHETIC_fit.jsonl \
  --metadata calibration/fixtures/SYNTHETIC_environment.json --min-samples 2 \
  --out runs/synthetic/profile.json
python -m calibration evaluate --profile runs/synthetic/profile.json \
  --traces calibration/fixtures/SYNTHETIC_holdout.jsonl --target-ms 11 \
  --out runs/synthetic/validation.json
```

Expected: 3 supported cells, 3 of 5 held-out steps covered, 1 target violation among 3 covered predictions deemed safe. One cell is deliberately undersampled and a separate row exceeds the configured decode range. The fitted 128-token upper time is raised to the 64-token envelope because the invented 128-token samples are unrealistically cheap. This exercises the monotonicity and fallback paths.
