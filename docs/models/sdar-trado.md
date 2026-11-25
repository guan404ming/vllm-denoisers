---
icon: model-org
organization: JetLM
---

# [![](../assets/orgs/JetLM.webp){ .org-avatar width="30" height="30" }](https://huggingface.co/collections/JetLM/sdar){ aria-label="SDAR models on Hugging Face" target="_blank" rel="noopener" } SDAR

[JetLM/SDAR](https://huggingface.co/collections/JetLM/sdar){ target="_blank" rel="noopener" }

## Serve

```bash
vllm serve JetLM/SDAR-1.7B-Chat --generation-config vllm
```

## Decoding

| Setting | Value |
| --- | --- |
| Policy | Confidence threshold. Tune `confidence_threshold` to accept more or fewer masked predictions. |
| Canvas | Four positions by default. Match the checkpoint’s trained block size. |

## Notes

- For `JetLM/SDAR-1.7B-Chat-b32`, pass `--hf-overrides '{"canvas_length":32}'`.
- The SDAR MoE adapter supports the 30B-A3B family.
