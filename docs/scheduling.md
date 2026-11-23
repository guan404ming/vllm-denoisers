# Request scheduling

With chunked prefill enabled, models that require canvas alignment use a lightweight scheduler by default. It keeps vLLM’s request ordering and collects no probe scores. Other models use the native scheduler.

Use the optional resident/active scheduler to give some decoding requests priority while others keep their canvas and cached keys and values in memory. The active limit controls how many resident decoding requests can advance on one step.

[![A and C advance this step. B and D wait with their canvases and cached keys and values in memory.](assets/resident-active-requests.svg){ width="100%" }](assets/resident-active-requests.svg)

## Enable shortest-first scheduling

```bash
vllm serve inclusionAI/LLaDA2.0-mini \
  --scheduler-cls resact \
  --attention-backend TRITON_ATTN --block-size 32 --max-num-seqs 32 \
  --max-num-batched-tokens 2048 --max-model-len 2048 --generation-config vllm
```

ResAct defaults to shortest-first ranking and an automatic active limit. This example allows up to 32 resident sequences. To reproduce fixed-limit scheduling, set `DENOISERS_ACTIVE=16`. For synchronous scheduling, add `--no-async-scheduling`.

Each scheduler starts with all decoding requests active. It measures completed decode-only steps at the model's configured canvas width. After 20 samples at both sizes, it can select a smaller observed batch if its median step is faster and retains at least 97% of the full batch’s decode rate. The rate is the number of requests advanced per step divided by median step time. The scheduler uses the latest 20 samples per size and retries the full batch every 20th eligible decision.

During prefill, with requests waiting for admission, or with a non-default canvas width, the scheduler lets all candidates advance. For overlapping steps, it measures the time between completions only when consecutive batches have the same size.

Requests keep priority after 100 ms without a scheduled step, 1.5 s without another output block, or 3.75 s from arrival without a first block. These protections can raise the active count above the target.

| Hold policy | How it ranks requests |
| --- | --- |
| `sjf` | Prefer a lower probe score. The default score uses mean block entropy. DiffusionGemma estimates remaining block steps from entropy and the current step. |
| `aging` | Increase priority with waiting time. Set `DENOISERS_AGE_S` to a positive time scale in seconds. |
| `random` | Shuffle candidates to provide a comparison without probe-based ordering. |

[![With two active slots, probe scores of 0.2, 0.5 and 0.8 put B and C ahead of A.](assets/shortest-first-ranking.svg){ width="100%" }](assets/shortest-first-ranking.svg)

Worker rank zero writes scores to a memory-mapped file. The scheduler reads that file. Run tensor-parallel ranks on the same node.

Set `DENOISERS_ACTIVE` to the configured `--max-num-seqs` to keep the probe active while letting every resident request advance. Use this control to measure the cost of scoring before comparing hold policies. See [Benchmarking](benchmarking.md).
