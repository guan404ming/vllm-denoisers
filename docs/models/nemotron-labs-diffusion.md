# Nemotron Labs Diffusion

## Serve

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve nvidia/Nemotron-Labs-Diffusion-3B \
  --attention-backend TRITON_ATTN --block-size 32 --max-model-len 2048 \
  --max-num-seqs 4 --max-num-batched-tokens 2048 --generation-config vllm
```

## Decoding

| Setting | Value |
| --- | --- |
| Block | 32 tokens. Causal prefill seeds the first block. Causal commit seeds the next. |
| Policy | Greedy, bidirectional denoising. Fill the most confident mask and all others at or above the threshold. |
| Threshold | `confidence_threshold=0.9` by default. |

## Notes

- For one token per step, pass `--hf-overrides '{"confidence_threshold":null}'`.
- Use the V2 model runner for text. See the [checkpoint example](https://huggingface.co/nvidia/Nemotron-Labs-Diffusion-3B#chat-with-our-model).
