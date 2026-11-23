# Fast-dLLM v2

## Serve

```bash
vllm serve Efficient-Large-Model/Fast_dLLM_v2_7B \
  --attention-backend TRITON_ATTN --block-size 32 --no-enable-chunked-prefill \
  --max-model-len 2048 --max-num-seqs 8 --max-num-batched-tokens 2048 --generation-config vllm
```

## Decoding

| Setting | Value |
| --- | --- |
| Policy | `fast_dllm` for 1.5B and 7B checkpoints. |
| Canvas | Match checkpoint `bd_size` (default 32). |
| Sub-block | `small_block_size` must divide the canvas length (default sub-block size 8). |

## Notes

- Seed each block with the previous block’s last prediction. Always pick the most likely token.
- At an exact prompt boundary, the first output block uses its own first position as the seed.
