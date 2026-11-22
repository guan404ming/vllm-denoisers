# DiffusionGemma

Ported from vLLM's native implementation onto the canvas runtime: entropy-bound accept, renoise, self-conditioning, stability check, causal commit forward.

## Sampler

- One Triton pass over bf16 logits computes argmax, entropy, and a categorical sample. The softcap is fused in; the sample is one inverse-CDF draw per row instead of Gumbel noise per element.
- Self-conditioning uses the tokens with probability at least 1e-3, at most 128 per position (about 96% of the mass), through `embedding_bag`. `--hf-overrides '{"sparse_self_conditioning": false}'` restores the dense `probs @ embed` product.
- The LM head uses vLLM's default. Select FP8 or INT4 through the [common model flags](models.md#common-flags).
- Commit rows skip the LM head.
- Logprobs come from the converging denoise step with the schedule temperature applied. The commit forward is causal and does not predict the canvas.
- Mean entropy for the confidence check covers only the positions that fit before `max_model_len`.

## Read extras

Per-request `extra_args`, as in vLLM:

| Key | Effect |
|---|---|
| `diffusion_canvas_length` | canvas width for this request |
| `diffusion_max_steps` | denoise step cap |
| `diffusion_seed_canvas` | initial canvas instead of random ids; length equals the width |
| `diffusion_pinned` | positions held at the seed on every step |
| `diffusion_read_only` | emit on convergence without a commit forward |
| `diffusion_constrained` | restrict tokens to `logprob_token_ids`; needs `logprobs == len(logprob_token_ids)` |

A constrained read masks the full logits instead of projecting only the allowed ids. Use `ignore_eos` when a read must fill the whole canvas.

## Images

Image embeddings go through the model state before self-conditioning. `max_num_batched_tokens` must be at least the image token count (2496).
