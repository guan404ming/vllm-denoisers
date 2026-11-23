# SDAR and TraDo

Serve a dense SDAR checkpoint with the confidence-threshold decoder.

```bash
vllm serve JetLM/SDAR-1.7B-Chat \
  --attention-backend TRITON_ATTN --block-size 16 --no-enable-chunked-prefill \
  --generation-config vllm
```

The adapter also registers the SDAR MoE architecture used by the 30B-A3B family. TraDo checkpoints such as `Gen-Verse/TraDo-8B-Instruct` use the same family.

The default canvas has four positions. Match the trained block size for other checkpoints. For `JetLM/SDAR-1.7B-Chat-b32`, set `--hf-overrides '{"canvas_length":32}'` and `--block-size 32`. The `confidence_threshold` override controls which masked predictions the decoder accepts in a step.
