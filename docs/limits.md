# Limits

- No pipeline parallelism, no batch-sharded sampling.
- A block that does not fit before `max_model_len` starts at the prompt end and decodes only the positions that fit.
- LLaDA2.2 block routing replaces the MoE router after construction; quantized or fused-routing MoE kernels and EPLB are rejected.
- Not supported: D2F, DiffuCoder, CoDA, RND1, Fast-dLLM v1.
