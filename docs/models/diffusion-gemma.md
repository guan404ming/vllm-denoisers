# DiffusionGemma

## Serve

```bash
VLLM_PLUGINS=denoisers vllm serve google/diffusiongemma-26B-A4B-it \
  --attention-backend TRITON_ATTN --max-num-seqs 8 --max-num-batched-tokens 4096
```

## Decoding

| Setting | Value |
| --- | --- |
| Policy | `entropy_bound` starts with random tokens, accepts confident positions, and renoises the rest. |
| Next-step input | Weighted token embeddings. Sparse mode uses probabilities ≥ 0.001, up to 128 tokens per position. |
| Commit | Causal attention. Logprobs come from the converging step at its scheduled temperature. |

## Notes

- For dense embeddings, pass `--hf-overrides '{"sparse_self_conditioning":false}'`. The model reads sampler defaults from its checkpoint.
- Budget enough batch tokens for each image (e.g., 2496 for a 2496-token image).
- Set the engine canvas limit with `--diffusion-config '{"canvas_length":32}'`. Pass request overrides in `SamplingParams.extra_args`.

## Request controls

| Key | Effect |
| --- | --- |
| `diffusion_canvas_length` | Set the request's canvas width within the engine limit. |
| `diffusion_max_steps` | Cap the denoise steps. |
| `diffusion_seed_canvas` | Start from these token IDs. Supply one ID per position. |
| `diffusion_pinned` | Keep the specified positions at their seed values. |
| `diffusion_read_only` | Emit tokens on convergence and skip the commit forward. |
| `diffusion_constrained` | Limit choices to `logprob_token_ids`. Set `logprobs` to the number of allowed IDs. |

For fixed-width reads, set `ignore_eos`.
