# Fast-dLLM v2

```bash
vllm serve Efficient-Large-Model/Fast_dLLM_v2_7B \
  --attention-backend TRITON_ATTN --block-size 32 --no-enable-chunked-prefill \
  --max-model-len 2048 --max-num-seqs 8 --max-num-batched-tokens 2048 --generation-config vllm
```

The 1.5B and 7B checkpoints use the `fast_dllm` policy. The canvas must match the checkpoint's `bd_size`. The `small_block_size` must divide the canvas length. The defaults are 32 and 8.

The decoder seeds the first position of a block from the previous block's last prediction. When the prompt ends exactly at a block boundary, the first output block uses its own first position for that seed. The seed always uses the most likely token.
