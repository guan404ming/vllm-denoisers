---
icon: model-org
organization: ByteDance-Seed
---

# [![](../assets/orgs/ByteDance-Seed.webp){ .org-avatar width="30" height="30" }](https://huggingface.co/collections/ByteDance-Seed/stable-diffcoder){ aria-label="Stable-DiffCoder models on Hugging Face" target="_blank" rel="noopener" } Stable-DiffCoder

[ByteDance-Seed/Stable-DiffCoder](https://huggingface.co/collections/ByteDance-Seed/stable-diffcoder){ target="_blank" rel="noopener" }

## Serve

```bash
vllm serve ByteDance-Seed/Stable-DiffCoder-8B-Instruct
```

## Decoding

| Setting | Value |
| --- | --- |
| Block | Four tokens, mask token ID 5. |
| Policy | Greedy. Fill the most confident mask each step, including a partial prompt block. |
| Threshold | Keep `confidence_threshold=1.0` for the checkpoint’s fixed transfer count. |

## Notes

- For base weights, use `ByteDance-Seed/Stable-DiffCoder-8B-Base`. [Checkpoint examples](https://github.com/ByteDance-Seed/Stable-DiffCoder).
