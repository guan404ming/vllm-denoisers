# Limits

- Attention: `TRITON_ATTN`, or `FLASH_ATTN` on Hopper and newer (per-request causal needs FA4) with [denoisers/patches/vllm-0.33.0rc1.dev487-flash-block-ranges.patch](../denoisers/patches/vllm-0.33.0rc1.dev487-flash-block-ranges.patch) applied to vLLM.
- No pipeline parallelism, no batch-sharded sampling.
- A KV policy turns off async scheduling. `prefix_look` needs prefix caching off.
- Tensor-parallel ranks must share one node when a denoisers scheduler is used.
- Chunked prefill needs a denoisers scheduler, except for DiffusionGemma.
- Prefix caching needs `--block-size` as a multiple of the canvas, except for DiffusionGemma. The prompt tail after the last canvas boundary is re-encoded with the first block.
- A block that does not fit before `max_model_len` starts at the prompt end and decodes only the positions that fit.
- LLaDA2.2 block routing replaces the MoE router after construction; quantized or fused-routing MoE kernels and EPLB are rejected.
- Full-attention models (LLaDA-8B, Dream) match the reference only while prompt and response fit in one canvas; see [deviations.md](deviations.md).
- Not supported: D2F, DiffuCoder, CoDA, RND1, Fast-dLLM v1.
