# vllm-denoisers

Serve diffusion language models through vLLM's OpenAI-compatible API. The plugin supplies model adapters and decoding policies. vLLM manages requests, batches, and the KV cache.

## Quick start

[Install the plugin](docs/installation.md) in your vLLM environment. Start LLaDA2 with the Triton attention backend.

```bash
vllm serve inclusionAI/LLaDA2.0-mini \
  --attention-backend TRITON_ATTN --block-size 32 --no-enable-chunked-prefill \
  --max-model-len 2048 --max-num-seqs 8 --max-num-batched-tokens 2048 \
  --generation-config vllm
```

In a second terminal, [send your first request](docs/quickstart.md#send-a-request). The generated text appears in `choices[0].text`.

## Documentation

Browse the [documentation index](docs/index.md) or choose a guide.

- [User guide](docs/user-guide.md) covers installation, supported checkpoints, and scheduling.
- [Benchmarking](docs/benchmarking.md) shows how to compare serving configurations.
- [Contributor guide](docs/contributor-guide.md) covers the runtime, policy extensions, and tests.
