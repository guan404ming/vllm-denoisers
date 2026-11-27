---
icon: model-org
organization: apple
---

# [![](../assets/orgs/apple.webp){ .org-avatar width="30" height="30" }](https://huggingface.co/apple/DiffuCoder-7B-cpGRPO){ aria-label="DiffuCoder on Hugging Face" target="_blank" rel="noopener" } DiffuCoder

[apple/DiffuCoder-7B-cpGRPO](https://huggingface.co/apple/DiffuCoder-7B-cpGRPO){ target="_blank" rel="noopener" }

## Serve

```bash
vllm serve apple/DiffuCoder-7B-cpGRPO \
  --no-enable-prefix-caching --max-model-len 2048 --generation-config vllm
```

## Decoding

| Setting | Value |
| --- | --- |
| Policy | `dream`: shifted logits, entropy-ranked unmasking. |
| Defaults | Greedy decoding, 256-position canvas. |
| Reference sampling | `--hf-overrides '{"temperature":0.4,"top_p":0.95}'` |

## Notes

- DiffuCoder declares `DreamModel`, so vLLM loads the [Dream](dream.md) adapter. [Checkpoint configuration](https://huggingface.co/apple/DiffuCoder-7B-cpGRPO/blob/main/config.json)
- The model fills the rest of the canvas with `<|dlm_pad|>` (id 151667) instead of an end token. Add `"stop_token_ids": [151667]` to each request, or the response runs to `max_tokens`.
- The checkpoint is licensed under the Apple AMLR license.
