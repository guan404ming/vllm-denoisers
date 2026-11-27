---
icon: model-org
organization: nvidia
---

# [![](../assets/orgs/nvidia.webp){ .org-avatar width="30" height="30" }](https://huggingface.co/collections/nvidia/nemotron-labs-sigma){ aria-label="Nemotron Labs Sigma models on Hugging Face" target="_blank" rel="noopener" } Nemotron Labs Sigma

[nvidia/Nemotron-Labs-Sigma](https://huggingface.co/collections/nvidia/nemotron-labs-sigma){ target="_blank" rel="noopener" }

## Serve

```bash
vllm serve nvidia/Nemotron-Labs-Sigma-distill-3B-nfe2 --trust-remote-code --dtype bfloat16
```

## Decoding

| Setting | Value |
| --- | --- |
| Base or instruct (3B, 8B) | `sigma`: DDIM, 32 log-SNR steps, guidance 3, score temperature 0.3. |
| Distilled (NFE 2, 4, 8, 16) | `sigma`: checkpoint step count, fresh noise per transition. |
| Distilled self-speculative | Autoregressive decoding with the AR head. |

## Notes

- Each block starts from noise and commits with causal attention. Guidance adds one block-only pass per step.
- The self-speculative reference returns the greedy output of the AR head, so the plugin serves these checkpoints as autoregressive models.
