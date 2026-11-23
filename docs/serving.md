# Batch and canvas limits

For models that use the shared token canvas, reserve one canvas per concurrent sequence.

[![Four 32-token canvases and two 64-token canvases each require a batch budget of at least 128 tokens.](assets/batch-token-budget.svg){ width="100%" }](assets/batch-token-budget.svg)

| Setting | Rule |
| --- | --- |
| Batch token budget | Set `--max-num-batched-tokens` to at least `max_num_seqs × canvas_length`. Lower `--max-num-seqs` to fit a larger canvas. |
| Prefix caching | For aligned policies, set `--block-size` and any explicit `--prefix-match-unit` to multiples of the canvas length. |
| Chunked prefill | With chunked prefill enabled, aligned policies automatically split intermediate chunks at canvas boundaries. The final chunk can be shorter. A custom scheduler must support this alignment. |

[![One 64-token cache block spans two complete 32-token canvases.](assets/cache-block-alignment.svg){ width="100%" }](assets/cache-block-alignment.svg)

## Shared runtime settings

- Pass model options with `--hf-overrides`. If checkpoint defaults fail request validation, pass `--generation-config vllm`.
- Masked-token samplers ignore request-level sampling seeds and penalties.
- Output logprobs come from the completed block with bidirectional commit, or the converging denoise step with causal commit.
- The token-canvas runtime supports tensor parallelism. It rejects pipeline parallelism and batch-sharded sampling.

## Output-head quantization

Set `lm_head_quantization` in `--hf-overrides` for an unquantized output head. Shared input embeddings retain their original weights.

| Format | Requirements |
| --- | --- |
| `fp8` | Ada or newer |
| `int4` | Ampere or newer. Input width divisible by 32 and local vocabulary shard size divisible by 8. |
