# Benchmarking

Use the same vLLM client and workload for each server configuration. The example measures DiffusionGemma serving latency under a fixed arrival rate. It requires access to the checkpoint and a GPU that fits the chosen batch.

## Start one server per run

In Bash, define the common server arguments. Keep these values fixed across runs.

```bash
model=google/diffusiongemma-26B-A4B-it
serve_args=("$model" --attention-backend TRITON_ATTN --max-num-seqs 32
  --max-model-len 4096 --max-num-batched-tokens 8192)
scheduler=denoisers.scheduler.resident_active.DenoisersAsyncScheduler
```

Select **one** launch command per run. Stop the previous server before starting the next. The empty plugin list selects native vLLM in this environment.

| Run name | Launch command |
| --- | --- |
| `native` | `VLLM_PLUGINS='' vllm serve "${serve_args[@]}"` |
| `plugin` | `VLLM_PLUGINS=denoisers vllm serve "${serve_args[@]}"` |
| `probe` | `VLLM_PLUGINS=denoisers DENOISERS_HOLD=sjf DENOISERS_ACTIVE=32 vllm serve "${serve_args[@]}" --scheduler-cls "$scheduler"` |
| `sjf` | `VLLM_PLUGINS=denoisers DENOISERS_HOLD=sjf DENOISERS_ACTIVE=16 vllm serve "${serve_args[@]}" --scheduler-cls "$scheduler"` |

Compare `native` with `plugin` to measure the runtime change. Compare `probe` with `sjf` to isolate the active limit.

## Send the workload

In a second terminal, set `run` to the server's run name. Use the same seed and prompts for all four runs.

```bash
run=native
vllm bench serve --model google/diffusiongemma-26B-A4B-it \
  --dataset-name random --random-input-len 128 --random-output-len 256 \
  --temperature 1 --ignore-eos --seed 1 --num-prompts 192 --request-rate 8 \
  --percentile-metrics ttft,e2el --metric-percentiles 50,99 --goodput e2el:20000 \
  --save-result --save-detailed --result-dir bench-results --result-filename "$run-seed1.json"
```

This workload uses synthetic inputs and fixes the output length. For a service workload, replay the same real prompts and let generation stop at EOS. Report the two workloads separately. See the [vLLM benchmark CLI](https://docs.vllm.ai/en/latest/cli/bench/serve/) for dataset and arrival options.

## Read the results

| Metric | Definition |
| --- | --- |
| Output throughput | Generated tokens per second across the run. |
| TTFT | Time from a request to its first streamed token. |
| End-to-end p50 and p99 | Median and 99th percentile of completed request latency. |
| SLO20 attainment | Successful requests that finish within 20 seconds divided by all measured requests sent. Count failed requests as misses. |
| Goodput | Requests per second that meet the configured latency target. |

For SLO20, use the detailed JSON `latencies` in seconds and the corresponding `errors`. For accuracy, run the paired GSM8K evaluation with natural stopping.

```bash
modal run tests/llada2_eval.py::gsm8k --model inclusionAI/LLaDA2.0-mini --n 100 --n-ref 30
```

See [Testing](testing.md) for Modal setup and cache handling. Report the full evaluation count and the paired reference count separately.

## Record a run

Save the raw JSON, both repository revisions, checkpoint revision, GPU model and count, dtype, server and client commands, and seed. Repeat each configuration with the same set of seeds. Record warmup settings and failures alongside the metrics.
