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

```bash
vllm serve nvidia/Nemotron-Labs-Sigma-3B-instruct \
  --trust-remote-code --enforce-eager --no-async-scheduling --no-enable-prefix-caching \
  --dtype bfloat16
```

## Decoding

| Setting | Value |
| --- | --- |
| Base or instruct (3B, 8B) | DDIM, 32 log-SNR steps, guidance 3, score temperature 0.3. |
| Distilled (NFE 2, 4, 8, 16) | `sigma` on the canvas runtime: checkpoint step count plus one, fresh noise per transition. |
| Distilled self-speculative | Diffusion drafts, greedy autoregressive verification. |

## Notes

- Distilled blockwise checkpoints run on the canvas runtime. Each block starts from fresh noise and commits with causal attention.
- Base, instruct, and self-speculative checkpoints use Model Runner V2 to prepare each block and return sampled token IDs. The pinned vLLM build selects V2 automatically for this single-GPU CUDA configuration with Triton installed.
- These checkpoints need one CUDA GPU, eager execution, unquantized weights, and default sampling parameters. Logprobs and per-request policy overrides are unsupported.
- The adapter retains latents, self-conditioning, and the clean prefix cache across steps. Each new block re-encodes the committed prefix. Self-speculative BF16 output may differ from an uninterrupted reference run at block boundaries.
