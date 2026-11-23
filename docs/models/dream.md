# Dream

## Serve

```bash
vllm serve Dream-org/Dream-v0-Instruct-7B \
  --attention-backend TRITON_ATTN --no-enable-prefix-caching --no-enable-chunked-prefill \
  --max-model-len 2048 --max-num-seqs 8 --max-num-batched-tokens 2048 --generation-config vllm
```

## Decoding

| Setting | Value |
| --- | --- |
| Policy | `dream`: shifted logits, entropy-ranked unmasking. |
| Defaults | Greedy decoding, 256-position canvas. |
| Reference sampling | `--hf-overrides '{"temperature":0.1,"top_p":0.9}'` |

## Notes

- Set `top_p` once in the model config. The sampler retains the top token and nucleus candidates with scaled probability ≥ 1/256.
- For reference parity, fit prompt and response on one canvas. Longer prompts retain only the trailing partial block. Longer responses span canvases.
- Set `denoise_steps` per full canvas.
