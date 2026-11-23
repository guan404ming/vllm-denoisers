# Nemotron Labs Diffusion

Serve the text adapter with the V2 model runner.

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve nvidia/Nemotron-Labs-Diffusion-3B \
  --attention-backend TRITON_ATTN --block-size 32 --max-model-len 2048 \
  --max-num-seqs 4 --max-num-batched-tokens 2048 --generation-config vllm
```

The causal prompt pass seeds the first token of a 32-token block. Bidirectional denoise steps fill the masked positions. A causal commit pass over the completed block seeds the next one.

The decoder uses greedy sampling. Each step fills the most confident masked position and every other masked position with confidence at or above `confidence_threshold`. The default is 0.9. Use `--hf-overrides '{"confidence_threshold":null}'` to fill one position per step. See [NVIDIA's diffusion example](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B#chat-with-our-model).
