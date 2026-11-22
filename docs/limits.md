# Limits

- Attention: `TRITON_ATTN`, or `FLASH_ATTN` on Hopper and newer (per-request causal needs FA4) with [patches/vllm-0.33.0rc1.dev487-flash-block-ranges.patch](../patches/vllm-0.33.0rc1.dev487-flash-block-ranges.patch) applied to vLLM.
- No pipeline parallelism, no batch-sharded sampling.
- Tensor-parallel ranks must share one node when a denoisers scheduler is used.
- Chunked prefill needs a denoisers scheduler, except for DiffusionGemma.
- Prefix caching needs `--block-size` as a multiple of the canvas, except for DiffusionGemma. The prompt tail after the last canvas boundary is re-encoded with the first block.
- A block that does not fit before `max_model_len` starts at the prompt end and decodes only the positions that fit.
- LLaDA2.2 block routing replaces the MoE router after construction; quantized or fused-routing MoE kernels and EPLB are rejected.
- Not supported by design: D2F, full-bidirectional models without a KV cache (LLaDA-8B, Dream, DiffuCoder, CoDA, RND1), Fast-dLLM v1.
