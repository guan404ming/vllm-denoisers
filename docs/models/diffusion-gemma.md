# DiffusionGemma

```bash
VLLM_PLUGINS=denoisers vllm serve google/diffusiongemma-26B-A4B-it \
  --attention-backend TRITON_ATTN --max-num-seqs 8 --max-num-batched-tokens 4096
```

The `entropy_bound` policy starts with random tokens, accepts confident positions, and adds fresh noise to the rest. It feeds a weighted token embedding into the next step. The sparse calculation uses tokens with probability at least 0.001, up to 128 per position. Set `--hf-overrides '{"sparse_self_conditioning":false}'` for the dense embedding calculation.

The commit pass uses causal attention. Output logprobs come from the converging denoise step at its scheduled temperature. To process images, allow at least 2496 tokens in the batch budget for an image of that token count.

## Control a request

Pass these keys in `SamplingParams.extra_args`. Use `--diffusion-config '{"canvas_length":32}'` to set an engine-wide canvas width.

| Key | Effect |
| --- | --- |
| `diffusion_canvas_length` | Set the request's canvas width within the engine limit. |
| `diffusion_max_steps` | Cap the denoise steps. |
| `diffusion_seed_canvas` | Start from these token IDs. Supply one ID per position. |
| `diffusion_pinned` | Keep the specified positions at their seed values. |
| `diffusion_read_only` | Emit tokens on convergence and skip the commit forward. |
| `diffusion_constrained` | Limit choices to `logprob_token_ids`. Set `logprobs` to the number of allowed IDs. |

Use `ignore_eos` for a read that must fill the requested width.
