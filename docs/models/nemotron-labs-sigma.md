# Nemotron Labs Sigma

Serve a checkpoint from the [NVIDIA Sigma collection](https://huggingface.co/collections/nvidia/nemotron-labs-sigma) on one CUDA GPU.

```bash
VLLM_USE_V2_MODEL_RUNNER=1 vllm serve nvidia/Nemotron-Labs-Sigma-distill-3B-nfe2 \
  --trust-remote-code --enforce-eager --no-async-scheduling --no-enable-prefix-caching \
  --dtype bfloat16
```

| Checkpoint | Sampler |
| --- | --- |
| 3B and 8B base or instruct | DDIM with 32 steps, log-SNR grid, guidance 3, score temperature 0.3 |
| Distilled NFE 2, 4, 8, or 16 | Fresh noise at each transition and the checkpoint's fixed step count |
| Distilled self-speculative | Diffusion drafts with greedy autoregressive verification |

vLLM schedules requests and returns completed blocks. For each request, the adapter calls the checkpoint sampler once per block and encodes the committed prefix again. Self-speculative BF16 output can differ at a block boundary from a single uninterrupted reference call.

Use the default request sampling parameters. This adapter requires eager execution and unquantized weights. It rejects logprobs and per-request policy overrides.
