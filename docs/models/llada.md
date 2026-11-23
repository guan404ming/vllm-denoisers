# LLaDA-8B

## Serve

```bash
vllm serve GSAI-ML/LLaDA-8B-Instruct \
  --attention-backend TRITON_ATTN --no-enable-prefix-caching --no-enable-chunked-prefill \
  --max-model-len 2048 --max-num-seqs 8 --max-num-batched-tokens 2048 --generation-config vllm
```

## Decoding

| Setting | Value |
| --- | --- |
| Policy | `llada`: sub-block offsets start at the response boundary. |
| Canvas | `canvas_length=256`, `sub_block_length=32` by default. |
| Step budget | `denoise_steps` spans the entire canvas. Set all three options via `--hf-overrides`. |

## Notes

- Fit prompt and response on one canvas for reference parity. Longer prompts prefill earlier tokens once. The trailing prompt block shares the canvas with the response.
- `max_tokens` can further limit the response. For a larger canvas, increase the [batch token budget](../serving.md).
- At positive temperatures, rank sampled tokens by their untempered probabilities.
