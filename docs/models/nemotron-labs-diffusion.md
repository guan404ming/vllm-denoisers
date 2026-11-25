---
icon: model-org
organization: nvidia
---

# [![](../assets/orgs/nvidia.webp){ .org-avatar width="30" height="30" }](https://huggingface.co/collections/nvidia/nemotron-labs-diffusion){ aria-label="Nemotron Labs Diffusion models on Hugging Face" target="_blank" rel="noopener" } Nemotron Labs Diffusion

[nvidia/Nemotron-Labs-Diffusion](https://huggingface.co/collections/nvidia/nemotron-labs-diffusion){ target="_blank" rel="noopener" }

## Serve

```bash
vllm serve nvidia/Nemotron-Labs-Diffusion-3B \
  --max-model-len 2048 --max-num-seqs 4 --generation-config vllm
```

## Image input

```bash
vllm serve nvidia/Nemotron-Labs-Diffusion-VLM-8B \
  --max-model-len 8192 --max-num-seqs 4 --max-num-batched-tokens 8192 --generation-config vllm
```

- Send `image_url` and `text` items in the user message’s `content` array to `POST /v1/chat/completions`. The model returns text.
- The Pixtral encoder processes the image. Each unseeded 32-token response block starts fully masked.

## Decoding

| Setting | Value |
| --- | --- |
| Block | 32 tokens. Causal prefill seeds the first block. Causal commit seeds the next. |
| Policy | Greedy, bidirectional denoising. Fill the most confident mask and all others at or above the threshold. |
| Threshold | `confidence_threshold=0.9` by default. |

## Notes

- For one token per step, pass `--hf-overrides '{"confidence_threshold":null}'`.
- See the [text checkpoint example](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B#chat-with-our-model).
