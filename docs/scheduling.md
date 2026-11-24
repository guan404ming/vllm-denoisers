# Request scheduling

With chunked prefill enabled, models that require canvas alignment use a lightweight scheduler by default. It keeps vLLM’s request ordering and collects no probe scores. Other models use the native scheduler.

Use the optional resident/active scheduler to give some decoding requests priority while others keep their canvas and cached keys and values in memory. The active limit controls how many resident decoding requests can advance on one step. ResAct ranks requests by the denoise steps they still need.

[![A and C advance this step. B and D wait with their canvases and cached keys and values in memory.](assets/resident-active-requests.svg){ width="100%" }](assets/resident-active-requests.svg)

## Enable shortest-first scheduling

```bash
DENOISERS_ACTIVE=16 vllm serve inclusionAI/LLaDA2.0-mini \
  --scheduler-cls resact \
  --attention-backend TRITON_ATTN --block-size 32 --max-num-seqs 32 \
  --max-num-batched-tokens 2048 --max-model-len 2048 --generation-config vllm
```

This example allows up to 32 resident sequences and advances at most 16 decoding requests per step. `DENOISERS_ACTIVE` defaults to 16. Prefill chunks align with the canvas. For synchronous scheduling, add `--no-async-scheduling`.

| Hold policy | How it ranks requests |
| --- | --- |
| `sjf` | Prefer fewer remaining steps. |
| `aging` | Increase priority with waiting time. Set `DENOISERS_AGE_S` to a positive time scale in seconds. |
| `random` | Shuffle candidates to provide a comparison without probe-based ordering. |

[![With two active slots, probe scores of 0.2, 0.5 and 0.8 put B and C ahead of A.](assets/shortest-first-ranking.svg){ width="100%" }](assets/shortest-first-ranking.svg)

## Remaining steps

The estimate has two parts. The probe score covers the current block. The default score uses mean block entropy. DiffusionGemma estimates remaining block steps from entropy and the current step. ResAct adds the steps that finished requests needed after the same block.

DiffusionGemma also reports when the current block predicts a stop token. A request with that signal has few steps left after the block, so it ranks ahead.

Each new block gets one step to produce a probe score. Until the score arrives, the request ranks at the mean score.

Worker rank zero writes scores to a memory-mapped file. The scheduler reads that file. Run tensor-parallel ranks on the same node.

[Benchmarking](benchmarking.md)
