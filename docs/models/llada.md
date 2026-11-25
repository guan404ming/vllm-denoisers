---
icon: model-org
organization: GSAI-ML
---

# [![](../assets/orgs/GSAI-ML.webp){ .org-avatar width="30" height="30" }](https://huggingface.co/models?author=GSAI-ML&search=LLaDA-8B){ aria-label="LLaDA-8B models on Hugging Face" target="_blank" rel="noopener" } LLaDA-8B

[GSAI-ML/LLaDA-8B](https://huggingface.co/models?author=GSAI-ML&search=LLaDA-8B){ target="_blank" rel="noopener" }

## Serve

```bash
vllm serve GSAI-ML/LLaDA-8B-Instruct \
  --no-enable-prefix-caching --max-model-len 2048
```

## Decoding

| Setting | Value |
| --- | --- |
| Policy | `llada`: sub-block offsets start at the response boundary. |
| Canvas | `canvas_length=256`, `sub_block_length=32` by default. |
| Step budget | `denoise_steps` spans the entire canvas. Set all three options via `--hf-overrides`. |

## Notes

- Fit prompt and response on one canvas for reference parity. Longer prompts prefill earlier tokens once. The trailing prompt block shares the canvas with the response. Splitting a longer prompt can change its bidirectional attention context.
- `max_tokens` can further limit the response. For a larger canvas, increase the [batch token budget](../serving.md).
- At positive temperatures, rank sampled tokens by their untempered probabilities.
