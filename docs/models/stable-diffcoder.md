# Stable-DiffCoder

## Serve

```bash
vllm serve ByteDance-Seed/Stable-DiffCoder-8B-Instruct \
  --attention-backend TRITON_ATTN --block-size 16 --no-enable-chunked-prefill \
  --generation-config vllm
```

## Decoding

| Setting | Value |
| --- | --- |
| Block | Four tokens, mask token ID 5. |
| Policy | Greedy. Fill the most confident mask each step, including a partial prompt block. |
| Threshold | Keep `confidence_threshold=1.0` for the checkpoint’s fixed transfer count. |

## Notes

- For base weights, use `ByteDance-Seed/Stable-DiffCoder-8B-Base`. [Checkpoint examples](https://github.com/ByteDance-Seed/Stable-DiffCoder).
