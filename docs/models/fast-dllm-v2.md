---
icon: model-org
organization: Efficient-Large-Model
---

# [![](../assets/orgs/Efficient-Large-Model.webp){ .org-avatar width="30" height="30" }](https://huggingface.co/collections/Efficient-Large-Model/fast-dllm){ aria-label="Fast-dLLM v2 models on Hugging Face" target="_blank" rel="noopener" } Fast-dLLM v2

[Efficient-Large-Model/Fast-dLLM](https://huggingface.co/collections/Efficient-Large-Model/fast-dllm){ target="_blank" rel="noopener" }

## Serve

```bash
vllm serve Efficient-Large-Model/Fast_dLLM_v2_7B \
  --attention-backend TRITON_ATTN --block-size 32 \
  --max-model-len 2048 --max-num-seqs 8 --max-num-batched-tokens 2048 --generation-config vllm
```

## Decoding

| Setting | Value |
| --- | --- |
| Policy | `fast_dllm` for 1.5B and 7B checkpoints. |
| Canvas | Match checkpoint `bd_size` (default 32). |
| Sub-block | `small_block_size` must divide the canvas length (default sub-block size 8). |

## Notes

- Seed each block with the previous block’s last prediction. Always pick the most likely token.
- At an exact prompt boundary, the first output block uses its own first position as the seed.
