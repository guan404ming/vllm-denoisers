---
icon: model-org
organization: inclusionAI
---

# [![](../assets/orgs/inclusionAI.webp){ .org-avatar width="30" height="30" }](https://huggingface.co/models?author=inclusionAI&search=LLaDA-MoE){ aria-label="LLaDA-MoE models on Hugging Face" target="_blank" rel="noopener" } LLaDA-MoE

[inclusionAI/LLaDA-MoE](https://huggingface.co/models?author=inclusionAI&search=LLaDA-MoE){ target="_blank" rel="noopener" }

## Serve

```bash
vllm serve inclusionAI/LLaDA-MoE-7B-A1B-Instruct \
  --no-enable-prefix-caching --max-model-len 2048 --generation-config vllm
```

## Decoding

| Setting | Value |
| --- | --- |
| Policy | `llada`, as for [LLaDA-8B](llada.md). |
| Canvas | `canvas_length=256`, `sub_block_length=32` by default. |

## Notes

- The checkpoint declares `model_type: "llada"`, like LLaDA-8B. The plugin selects the Qwen3-MoE layers when `architectures` names `LLaDAMoEModel`.
- Fit prompt and response on one canvas for reference parity.
