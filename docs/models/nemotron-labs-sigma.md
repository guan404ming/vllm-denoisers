# Nemotron Labs Sigma

## Serve

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve nvidia/Nemotron-Labs-Sigma-distill-3B-nfe2 \
  --trust-remote-code --enforce-eager --no-async-scheduling --no-enable-prefix-caching \
  --dtype bfloat16
```

## Decoding

| Setting | Value |
| --- | --- |
| Base or instruct (3B, 8B) | DDIM, 32 log-SNR steps, guidance 3, score temperature 0.3. |
| Distilled (NFE 2, 4, 8, 16) | Fixed checkpoint step count, fresh noise per transition. |
| Distilled self-speculative | Diffusion drafts, greedy autoregressive verification. |

## Notes

- Use one CUDA GPU, eager execution, unquantized weights, and default sampling parameters. Logprobs and per-request policy overrides are unsupported.
- Each block re-encodes the committed prefix. Self-speculative BF16 output may differ from an uninterrupted reference run at block boundaries. [Checkpoint collection](https://huggingface.co/collections/nvidia/nemotron-labs-sigma).
