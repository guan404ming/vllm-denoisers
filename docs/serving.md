# Batch and canvas limits

For models that use the shared token canvas, reserve one canvas per concurrent sequence.

| Setting | Rule |
| --- | --- |
| Batch token budget | Set `--max-num-batched-tokens` to at least `max_num_seqs × canvas_length`. Lower `--max-num-seqs` to fit a larger canvas. |
| Prefix caching | For aligned policies, set `--block-size` and any explicit `--prefix-match-unit` to multiples of the canvas length. |
| Chunked prefill | For aligned policies, pass `--no-enable-chunked-prefill` or use a [canvas-aware scheduler](scheduling.md). |

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
