# Request scheduling

The scheduler chooses which requests advance at each decoding step.

## Choose a scheduler

For automatic selection, omit `--scheduler-cls`.

| `--scheduler-cls` | Behavior |
| --- | --- |
| omitted | Align prefill chunks when required. Otherwise use vLLM’s default. |
| `aligned` | Align prefill chunks and keep vLLM’s request order. |
| `resact` | Apply the active limit and rank resident requests with a hold policy. |

- Both names follow vLLM’s scheduling mode. Add `--no-async-scheduling` for synchronous execution.
- For a custom scheduler, pass its full class path.
- The `aligned` scheduler collects no probe scores.

## Resident/active scheduling

Use the optional resident/active scheduler to give some decoding requests priority while others keep their canvas and cached keys and values in memory. `--max-num-seqs` sets how many requests stay resident. `DENOISERS_ACTIVE` sets how many of them advance on one step.

[![A and C advance this step. B and D wait with their canvases and cached keys and values in memory.](../assets/resident-active-requests.svg){ width="100%" }](../assets/resident-active-requests.svg)

```bash
DENOISERS_ACTIVE=16 vllm serve inclusionAI/LLaDA2.0-mini \
  --scheduler-cls resact --max-num-seqs 32 \
  --max-model-len 2048 --generation-config vllm
```

This example keeps up to 32 requests resident and advances 16 on each step. `DENOISERS_RESERVE` (default 0.125) gives that share of the steps to the oldest requests. Prefill chunks align with the canvas. For synchronous scheduling, add `--no-async-scheduling`. Keep `--max-num-batched-tokens` above `DENOISERS_ACTIVE × canvas_length`, so that new requests can prefill while the active requests decode.

## Hold policies

With the resident/active scheduler, set `DENOISERS_HOLD` to choose a hold policy. For example, the `aging` policy increases priority with wait time:

```bash
DENOISERS_HOLD=aging DENOISERS_AGE_S=2 vllm serve <model> --scheduler-cls resact
```

| Hold policy | How it ranks requests |
| --- | --- |
| `sjf` | Prefer fewer expected remaining steps. This is the default. |
| `aging` | Increase priority with waiting time. Set `DENOISERS_AGE_S` to a positive time scale in seconds. |
| `random` | Shuffle candidates to provide a comparison without probe-based ordering. |

[![With two active slots, probe scores of 0.2, 0.5 and 0.8 put B and C ahead of A.](../assets/shortest-first-ranking.svg){ width="100%" }](../assets/shortest-first-ranking.svg)

Worker rank zero writes scores to a memory-mapped file. The scheduler reads that file. Run tensor-parallel ranks on the same node.

[Write a hold policy](../extending.md#hold-policy) · [Benchmarking](../benchmarking.md)
