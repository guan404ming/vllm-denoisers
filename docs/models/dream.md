# Dream

```bash
vllm serve Dream-org/Dream-v0-Instruct-7B \
  --attention-backend TRITON_ATTN --no-enable-prefix-caching --no-enable-chunked-prefill \
  --max-model-len 2048 --max-num-seqs 8 --max-num-batched-tokens 2048 --generation-config vllm
```

The `dream` decoder uses shifted logits and entropy ranking. It defaults to greedy decoding on a 256-position canvas. Set `--hf-overrides '{"temperature":0.1,"top_p":0.9}'` for the reference sampling settings. Supply `top_p` once in the model config. The sampler keeps the most likely token and considers other nucleus tokens with scaled probability of at least 1/256.

Size the canvas to fit both prompt and response when comparing with the Hugging Face loop. A longer prompt leaves only its trailing partial block on the canvas. A longer response spans additional canvases. Use `denoise_steps` to set the step budget for a full canvas.
