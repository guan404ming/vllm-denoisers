# <img src="docs/assets/denoisers.svg" height="28" alt=""> vllm-denoisers

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)

Serve diffusion language models through vLLM’s OpenAI-compatible API. Each request refines a block over several steps, then emits the completed tokens.

Supported families include LLaDA2, Dream, DiffusionGemma, and Nemotron Labs Sigma. [Full model list](docs/models.md)

## Quick start

On a Linux CUDA host, install the [compatible vLLM build](docs/installation.md). From the repository root, install the plugin and serve LLaDA2.

```bash
uv pip install -e .
vllm serve inclusionAI/LLaDA2.0-mini \
  --attention-backend TRITON_ATTN --block-size 32 --no-enable-chunked-prefill \
  --max-model-len 2048 --max-num-seqs 8 --max-num-batched-tokens 2048 \
  --generation-config vllm
```

[Send your first request](docs/quickstart.md#send-a-request) · [Serving settings](docs/serving.md) · [Benchmarking](docs/benchmarking.md) · [Documentation](docs/index.md)
