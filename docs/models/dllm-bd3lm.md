---
icon: model-org
organization: dllm-hub
---

# [![](../assets/orgs/dllm-hub.webp){ .org-avatar width="30" height="30" }](https://huggingface.co/collections/dllm-hub/tiny-a2d){ aria-label="Tiny-A2D models on Hugging Face" target="_blank" rel="noopener" } dLLM Tiny-A2D

[dllm-hub/Tiny-A2D](https://huggingface.co/collections/dllm-hub/tiny-a2d){ target="_blank" rel="noopener" }

## Serve

```bash
vllm serve dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1 --max-model-len 2048 --generation-config vllm
```

```bash
vllm serve dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1 \
  --no-enable-prefix-caching --max-model-len 2048 --generation-config vllm
```

## Decoding

| Setting | Value |
| --- | --- |
| BD3LM | `low_confidence`: 32-token canvas with block-causal attention, one token per step. |
| MDLM | `llada`: 256-token canvas with 64-token sub-blocks, one token per step. |

## Notes

- Each MDLM and BD3LM pair shares one configuration. The plugin selects MDLM decoding when the model name contains `mdlm`. For a local copy without that name, pass `--hf-overrides '{"decode_policy":"llada","canvas_length":256,"sub_block_length":64}'`.
- Qwen3 checkpoints load the SDAR adapter. Qwen2.5-Coder checkpoints load the Qwen2 adapter of [Fast-dLLM v2](fast-dllm-v2.md).
- Disable thinking in the Qwen3 chat template (`enable_thinking=False`), as the model cards do.
- Fit MDLM prompt and response on one canvas for reference parity, as for [LLaDA-8B](llada.md).
