# Request scheduling

With chunked prefill enabled, models that require canvas alignment use a lightweight scheduler by default. It keeps vLLM’s request ordering and collects no probe scores. Other models use the native scheduler.

Use the optional resident/active scheduler to give some decoding requests priority while others keep their canvas and cached keys and values in memory. The active limit controls how many resident decoding requests can advance on one step.

[![A and C advance this step. B and D wait with their canvases and cached keys and values in memory.](assets/resident-active-requests.svg){ width="100%" }](assets/resident-active-requests.svg)

## Enable shortest-first scheduling

```bash
DENOISERS_HOLD=sjf DENOISERS_ACTIVE=16 vllm serve inclusionAI/LLaDA2.0-mini \
  --scheduler-cls resact \
  --attention-backend TRITON_ATTN --block-size 32 --max-num-seqs 32 \
  --max-num-batched-tokens 2048 --max-model-len 2048 --generation-config vllm
```

This example allows up to 32 sequences and advances at most 16 decoding requests per step. Prefill chunks align with the canvas. For synchronous scheduling, add `--no-async-scheduling`.

| Hold policy | How it ranks requests |
| --- | --- |
| `sjf` | Prefer a lower probe score. The default score uses mean block entropy. DiffusionGemma estimates remaining block steps from entropy and the current step. |
| `aging` | Increase priority with waiting time. Set `DENOISERS_AGE_S` to a positive time scale in seconds. |
| `random` | Shuffle candidates to provide a comparison without probe-based ordering. |

[![With two active slots, probe scores of 0.2, 0.5 and 0.8 put B and C ahead of A.](assets/shortest-first-ranking.svg){ width="100%" }](assets/shortest-first-ranking.svg)

Worker rank zero writes scores to a memory-mapped file. The scheduler reads that file. Run tensor-parallel ranks on the same node.

Set `DENOISERS_ACTIVE` to the configured `--max-num-seqs` to keep the probe active while letting every resident request advance. Use this control to measure the cost of scoring before comparing hold policies. See [Benchmarking](benchmarking.md).
