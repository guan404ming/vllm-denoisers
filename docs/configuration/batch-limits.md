# Batch limits

For models that use the shared token canvas, each concurrent sequence needs one canvas in the batch. Three vLLM settings must fit the canvas length of the checkpoint. Each [recipe](../models.md) already uses values that fit.

## Batch token budget

[![Four 32-token canvases and two 64-token canvases each require a batch budget of at least 128 tokens.](../assets/batch-token-budget.svg){ width="100%" }](../assets/batch-token-budget.svg)

The plugin serves at most `max_num_batched_tokens ÷ canvas_length` sequences at a time. At startup it lowers `max_num_seqs` to that number and logs the change.

To serve more sequences, raise `--max-num-batched-tokens`:

```bash
vllm serve google/diffusiongemma-26B-A4B-it --max-num-batched-tokens 4096
```

| Canvas length | `--max-num-batched-tokens` | Concurrent sequences |
| --- | --- | --- |
| 32 | 2048 | 64 |
| 32 | 8192 | 256 |
| 256 | 4096 | 16 |
| 256 | 8192 | 32 |

- If you omit the flag, vLLM chooses a default for the GPU.
- The budget must hold at least one canvas.
- To serve fewer sequences, set `--max-num-seqs`.
- With `--scheduler-cls resact`, the budget limits `DENOISERS_ACTIVE` instead. `--max-num-seqs` sets the resident count and can exceed the budget.

## Prefix caching

[![One 64-token cache block spans two complete 32-token canvases.](../assets/cache-block-alignment.svg){ width="100%" }](../assets/cache-block-alignment.svg)

The recipes need no `--block-size`. If you set `--block-size` or `--prefix-match-unit` with an aligned policy, use a multiple of the canvas length. For a 32-token canvas, 32 and 64 are valid. Any other value stops the server at startup.

```bash
vllm serve inclusionAI/LLaDA2.0-mini --block-size 32 \
  --max-model-len 2048 --generation-config vllm
```

To turn prefix caching off, pass `--no-enable-prefix-caching`.

## Chunked prefill

Chunked prefill needs no setting. Aligned policies split intermediate chunks at canvas boundaries, and the final chunk can be shorter. A custom scheduler must support this alignment. If it does not, pass `--no-enable-chunked-prefill`.

## Shared runtime settings

- Pass model options with `--hf-overrides`. If checkpoint defaults fail request validation, pass `--generation-config vllm`.
- Masked-token samplers ignore request-level sampling seeds and penalties.
- Output logprobs come from the completed block with bidirectional commit, or the converging denoise step with causal commit.
- The token-canvas runtime supports tensor parallelism. It rejects pipeline parallelism and batch-sharded sampling.
