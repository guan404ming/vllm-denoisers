# Model guide

| Family | Example checkpoint | Decoding |
| --- | --- | --- |
| [LLaDA2](models/llada2.md) | `inclusionAI/LLaDA2.0-mini` | Confidence, joint editing, or insert/delete |
| [SDAR and TraDo](models/sdar-trado.md) | `JetLM/SDAR-1.7B-Chat` | Confidence threshold |
| [Fast-dLLM v2](models/fast-dllm-v2.md) | `Efficient-Large-Model/Fast_dLLM_v2_7B` | Seeded sub-blocks |
| [Stable-DiffCoder](models/stable-diffcoder.md) | `ByteDance-Seed/Stable-DiffCoder-8B-Instruct` | Confidence-ranked unmasking |
| [LLaDA-8B](models/llada.md) | `GSAI-ML/LLaDA-8B-Instruct` | Full attention within a canvas |
| [Dream](models/dream.md) | `Dream-org/Dream-v0-Instruct-7B` | Entropy-ranked unmasking |
| [DiffusionGemma](models/diffusion-gemma.md) | `google/diffusiongemma-26B-A4B-it` | Entropy-bound acceptance and renoising |
| [Nemotron Labs Diffusion](models/nemotron-labs-diffusion.md) | `nvidia/Nemotron-Labs-Diffusion-3B` | Greedy text decoding with causal block seeds |
| [Nemotron Labs Sigma](models/nemotron-labs-sigma.md) | `nvidia/Nemotron-Labs-Sigma-distill-3B-nfe2` | Checkpoint sampler |

## Configure the shared canvas runtime

- Set `--max-num-batched-tokens` to at least `max_num_seqs × canvas_length`. Reduce `--max-num-seqs` to fit a larger canvas.
- For an aligned policy, set `--block-size` and any explicit `--prefix-match-unit` to multiples of the canvas length when prefix caching is enabled. Use `--no-enable-chunked-prefill` or select a [canvas-aware scheduler](scheduling.md).
- Use `--hf-overrides` for model-specific decoding options. For checkpoints with generation defaults that conflict with vLLM diffusion request validation, pass `--generation-config vllm`. DiffusionGemma reads its checkpoint's sampler config.
- The masked-token samplers ignore request-level sampling seeds and penalties. For policies with a bidirectional commit, output logprobs come from the completed block. Policies with a causal commit use the converging denoise step.
- The shared runtime supports tensor parallelism. It rejects pipeline parallelism and batch-sharded sampling.

## Reuse KV or quantize the output head

Select `--hf-overrides '{"kv_policy":"masked"}'` to recompute positions that were masked before the previous update. Other positions reuse cached keys and values until a full pass. Use `--no-async-scheduling` with a KV policy. Set `kv_refresh` to a positive step interval for periodic full passes. The [KV policy reference](extending.md#kv-policy) also describes windowed updates.

Set `lm_head_quantization` to `fp8` on Ada or newer GPUs, or `int4` on Ampere or newer GPUs, to quantize an unquantized output head. Shared input embeddings keep their original weights. INT4 requires an input width divisible by 32 and local vocabulary shards divisible by eight.
