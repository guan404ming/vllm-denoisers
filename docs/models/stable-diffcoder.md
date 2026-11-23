# Stable-DiffCoder

```bash
vllm serve ByteDance-Seed/Stable-DiffCoder-8B-Instruct \
  --attention-backend TRITON_ATTN --block-size 16 --no-enable-chunked-prefill \
  --generation-config vllm
```

Use `ByteDance-Seed/Stable-DiffCoder-8B-Base` for the base checkpoint. Both checkpoints use four-token blocks and mask token ID 5. The default greedy policy fills the most confident masked position on each step, including when the prompt ends inside a block. Keep `confidence_threshold` at its default of 1.0 for the fixed transfer count in the [checkpoint examples](https://github.com/ByteDance-Seed/Stable-DiffCoder).
