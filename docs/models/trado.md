---
icon: model-org
organization: Gen-Verse
---

# [![](../assets/orgs/Gen-Verse.webp){ .org-avatar width="30" height="30" }](https://huggingface.co/collections/Gen-Verse/trado-series){ aria-label="TraDo models on Hugging Face" target="_blank" rel="noopener" } TraDo

[Gen-Verse/TraDo Series](https://huggingface.co/collections/Gen-Verse/trado-series){ target="_blank" rel="noopener" }

## Serve

```bash
vllm serve Gen-Verse/TraDo-8B-Instruct --generation-config vllm
```

## Decoding

| Setting | Value |
| --- | --- |
| Policy | `low_confidence` |
| Plugin defaults | Four-token canvas, `confidence_threshold=0.85` |

## Notes

- TraDo-8B-Instruct declares `SDARForCausalLM`, so vLLM loads the SDAR adapter. [Checkpoint configuration](https://huggingface.co/Gen-Verse/TraDo-8B-Instruct/blob/main/config.json)
