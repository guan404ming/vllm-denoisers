# Models

## Common flags

- `--attention-backend`: `TRITON_ATTN` works on stock vLLM. `FLASH_ATTN` (Hopper and newer; per-request causal needs FA4) needs [denoisers/patches/vllm-0.33.0rc1.dev487-flash-block-ranges.patch](../denoisers/patches/vllm-0.33.0rc1.dev487-flash-block-ranges.patch); stock vLLM drops the prefill block ranges under a per-request causal mask and runs prefill as plain causal (LLaDA2.0-mini GSM8K 91.9% to 87.9%).
- `--block-size`: a multiple of the canvas (32 for LLaDA2, 4 for SDAR) so prefix caching works. The prompt tail after the last canvas boundary is re-encoded with the first block.
- `--no-enable-chunked-prefill`, unless a denoisers scheduler is used.
- `--generation-config vllm`: when `generation_config.json` sets a temperature. vLLM rejects diffusion requests whose temperature is not 1.0.
- `--quantization fp8_per_tensor`: about +30% on dense SDAR at canvas 32, same GSM8K score.
- `--compilation-config '{"max_cudagraph_capture_size": N}'` with N at least `max_num_seqs * canvas_length`: a step forwards batch times canvas tokens, so past vLLM's default of 512 it runs eagerly. Small models gain most.
- `--hf-overrides '{"lm_head_quantization": "fp8"}'`: quantizes a copy of an unquantized LM head to FP8 weights and activations on Ada and newer; shared input embeddings keep their weights. +7% on SDAR-8B at canvas 4, no gain on LLaDA2.0-mini.
- `"lm_head_quantization": "int4"`: packed INT4 weights with BF16 activations on Ampere and newer. The input width must be divisible by 32 and each local vocabulary shard a multiple of eight rows.
- `--hf-overrides '{"kv_policy": "masked"}'`: reuses the K/V of unmasked cells inside a block; see [extending.md](extending.md). It turns off async scheduling, which every other configuration keeps on (+24% on SDAR-8B at batch 32). LLaDA-8B commits left to right and works with `kv_refresh: 32`; Dream commits anywhere in the canvas and needs `kv_refresh: 8`. See [results.md](results.md).

## LLaDA2 (BailingMoeV2)

| Checkpoint | Policy | Note |
|---|---|---|
| `LLaDA2.0-mini`, `-flash` | `low_confidence` | default |
| `LLaDA2.1-mini`, `-flash` | `joint` | `--hf-overrides '{"decode_policy": "joint"}'` |
| `LLaDA2.2-mini`, `-flash` | `indel` | chosen from `expert_capacity`; block routing |

## Stable-DiffCoder (Llama)

The 8B Base and Instruct checkpoints use four-token blocks and mask ID 5. Greedy decoding fills the most confident masked position on each step, including when the prompt ends inside a block. The default threshold of 1.0 reproduces the fixed transfer count in the [official examples](https://github.com/ByteDance-Seed/Stable-DiffCoder).

```bash
vllm serve ByteDance-Seed/Stable-DiffCoder-8B-Instruct \
  --attention-backend TRITON_ATTN --block-size 16 --no-enable-chunked-prefill
```

## SDAR and TraDo (Qwen3)

- `JetLM/SDAR-{1.7B,4B,8B}-Chat[-b8,-b16,-b32,-b64]`, `JetLM/SDAR-30B-A3B-Chat`, `-Sci`, `Gen-Verse/TraDo-{4B,8B}-Instruct`, `TraDo-8B-Thinking`.
- Canvas 4 by default. `--hf-overrides '{"canvas_length": 32}'` for the `-b32` checkpoints.

## Fast-dLLM v2 (Qwen2.5)

- `Efficient-Large-Model/Fast_dLLM_v2_{1.5B,7B}`, policy `fast_dllm`.
- Shifted logits, sub-blocks of `small_block_size`.

## LLaDA-8B (Llama)

- `GSAI-ML/LLaDA-8B-Instruct`, policy `llada`, canvas 256 unmasked one token a step in sub-blocks of `sub_block_length` 32, greedy. The sub-block is the reference `block_length`.
- The prompt tail shares the canvas, as for Dream. Set `canvas_length` to about the prompt plus the reference `gen_length`: the masks after the prompt are the response, and LLaDA plans its answer for the masks it sees. `--max-num-batched-tokens` must cover `max_num_seqs * canvas_length`.
- `denoise_steps` counts over the whole canvas. For a reference run with `gen_length` 512, `block_length` 32, 128 steps, and temperature 0.2, use `--hf-overrides '{"canvas_length": 768, "sub_block_length": 32, "denoise_steps": 192, "temperature": 0.2}'`.

## Dream (Qwen2.5)

- `Dream-org/Dream-v0-Instruct-7B`, policy `dream`, canvas 256, `alg=entropy`, greedy.
- The reference evaluation samples with `--hf-overrides '{"temperature": 0.1, "top_p": 0.9}'`. Greedy decoding answers tersely and scores much lower on GSM8K.
- Shifted logits like Fast-dLLM v2. `--hf-overrides '{"denoise_steps": 128}'` for two tokens a step; `canvas_length` sets the generation length of one block.
- A prompt shorter than the canvas sits in the canvas and is recomputed with the response every step, as in the HF loop; the steps scale with the masked part, so `denoise_steps` stays the per-canvas density. Set `--hf-overrides '{"canvas_length": 1024}'` (with `--max-num-batched-tokens` at least `max_num_seqs * 1024`) when prompt plus response exceed 256 tokens.

## DiffusionGemma (Gemma4 MoE)

- `google/diffusiongemma-26B-A4B-it`, policy `entropy_bound`, canvas 256 from the HF config.
- `--diffusion-config '{"canvas_length": 32}'` changes the canvas.
- Images, prefix caching, chunked prefill, logprobs, read extras, and TP work. See [diffusion_gemma.md](diffusion_gemma.md).

## Nemotron Labs Diffusion

The text adapter uses greedy diffusion with 32-token blocks. The causal prompt seeds the first block. Bidirectional attention fills masked positions. A causal pass over the completed block seeds the next one.

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve nvidia/Nemotron-Labs-Diffusion-3B \
  --attention-backend TRITON_ATTN --block-size 32 --max-model-len 2048 \
  --max-num-seqs 4 --max-num-batched-tokens 2048 --generation-config vllm
```

Set `confidence_threshold` through `--hf-overrides`. The default is 0.9. Each step fills all masked positions at or above the threshold and always fills the most confident position. Set the threshold to `null` for one position per step. See [NVIDIA's diffusion example](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B#chat-with-our-model).

## Nemotron Labs Sigma

Serve a checkpoint from the [NVIDIA Sigma collection](https://huggingface.co/collections/nvidia/nemotron-labs-sigma) on one CUDA GPU.

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve nvidia/Nemotron-Labs-Sigma-distill-3B-nfe2 \
  --trust-remote-code --enforce-eager --no-async-scheduling --no-enable-prefix-caching
```

| Checkpoint | Sampling recipe |
|---|---|
| 3B/8B base and instruct | DDIM, 32 steps, log-SNR grid, guidance 3, score temperature 0.3 |
| Distilled NFE 2/4/8/16 | Fresh noise at each transition and the checkpoint's fixed step count |
| Distilled self-speculative | Diffusion draft and greedy AR verification from the checkpoint |

vLLM schedules requests and streams completed blocks. The adapter calls NVIDIA's PyTorch sampler once per request per block. Each call encodes the committed prefix again. With BF16, self-speculative output can diverge at a block boundary from a single uninterrupted reference call. Use BF16 and the default sampling parameters. Logprobs, custom decode policies, tensor parallelism, and CUDA graphs are unsupported.

## Decode knobs

`--hf-overrides` accepts `decode_policy`, `temperature`, `top_p` (Dream), `denoise_steps`, `confidence_threshold`, `editing_threshold`, `max_post_steps`, `max_steps_per_block`, `sub_block_length`, `kv_policy`, `kv_refresh`, `prefix_look`, `after_look`, `input_policy`, `smooth_weight`, `smooth_initial`, `smooth_growth`. The policy keys also take a qualified class name; see [extending.md](extending.md).
