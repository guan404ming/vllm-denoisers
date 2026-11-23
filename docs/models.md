# Model guide

## Token-canvas models

- [DiffusionGemma](models/diffusion-gemma.md) · Confidence-based acceptance, renoising, and soft inputs.
- [Dream](models/dream.md) · Entropy-ranked unmasking with shifted logits.
- [Fast-dLLM v2](models/fast-dllm-v2.md) · Greedy block seeds and sub-block decoding.
- [LLaDA-8B](models/llada.md) · Full-canvas attention with sub-block unmasking.
- [LLaDA2](models/llada2.md) · Confidence thresholds, joint editing, or insert/delete decoding.
- [Nemotron Labs Diffusion](models/nemotron-labs-diffusion.md) · Greedy unmasking with causal block seeds.
- [SDAR and TraDo](models/sdar-trado.md) · Confidence-threshold decoding.
- [Stable-DiffCoder](models/stable-diffcoder.md) · Confidence-ranked unmasking in four-token blocks.

## Continuous-latent models

- [Nemotron Labs Sigma](models/nemotron-labs-sigma.md) · Latent diffusion with checkpoint-specific sampling or draft verification.

## Image understanding

Serve either checkpoint with the V2 model runner.

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve nvidia/Nemotron-Labs-Diffusion-VLM-8B \
  --attention-backend TRITON_ATTN --block-size 32 --no-enable-chunked-prefill \
  --max-model-len 8192 --max-num-seqs 4 --max-num-batched-tokens 8192 --generation-config vllm
```

For LLaDA, set the model to `inclusionAI/LLaDA2.0-Uni`. Send an `image_url` and a text question to `/v1/chat/completions`. Both models return text.

- Nemotron uses a Pixtral vision encoder. Each unseeded 32-token response block starts fully masked. See the [NVIDIA checkpoint](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-VLM-8B).
- LLaDA2.0-Uni accepts one image per request. Its visual tokenizer converts patches to discrete codes before block diffusion. Keep the checkpoint's `image_tokenizer/` directory when serving local weights. See the [inclusionAI checkpoint](https://huggingface.co/inclusionAI/LLaDA2.0-Uni).
