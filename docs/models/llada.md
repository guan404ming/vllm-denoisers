# LLaDA-8B

```bash
vllm serve GSAI-ML/LLaDA-8B-Instruct \
  --attention-backend TRITON_ATTN --no-enable-prefix-caching --no-enable-chunked-prefill \
  --max-model-len 2048 --max-num-seqs 8 --max-num-batched-tokens 2048 --generation-config vllm
```

The `llada` policy uses a 256-position canvas and 32-position sub-blocks by default. It starts each sub-block at an offset from the response start. With a positive decoding temperature, it ranks sampled tokens by their untempered probabilities.

Choose a canvas that fits the prompt and the intended response. A longer prompt leaves only its trailing partial block on the canvas. The model prefills earlier tokens once, which changes their interaction with the response. The available response region is the canvas minus the prompt tail. The request's `max_tokens` can truncate that region.

Set `canvas_length`, `sub_block_length`, and `denoise_steps` through `--hf-overrides`. The step budget covers the whole canvas. Increase the [batch token budget](../models.md#configure-the-shared-canvas-runtime) when you increase the canvas.
