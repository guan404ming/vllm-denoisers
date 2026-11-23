# SDAR and TraDo

## Serve

```bash
vllm serve JetLM/SDAR-1.7B-Chat \
  --attention-backend TRITON_ATTN --block-size 16 --no-enable-chunked-prefill \
  --generation-config vllm
```

## Decoding

| Setting | Value |
| --- | --- |
| Policy | Confidence threshold. Tune `confidence_threshold` to accept more or fewer masked predictions. |
| Canvas | Four positions by default. Match the checkpoint’s trained block size. |

## Notes

- For `JetLM/SDAR-1.7B-Chat-b32`, pass `--hf-overrides '{"canvas_length":32}' --block-size 32`.
- The same adapter supports SDAR MoE (30B-A3B) and TraDo (e.g., `Gen-Verse/TraDo-8B-Instruct`).
