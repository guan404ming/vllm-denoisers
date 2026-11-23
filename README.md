# <sub><img src="docs/assets/denoisers.svg" width="44" height="44" alt=""></sub>&#x2004;vllm-denoisers

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)

Serve diffusion language models through vLLM’s OpenAI-compatible API. Each request refines a block over several steps, then emits the completed tokens.

Supported families include LLaDA2, Dream, DiffusionGemma, and Nemotron Labs Sigma.

[Documentation](https://guan404ming.github.io/vllm-denoisers/) · [Full model list](docs/models.md)

## Quick start

On a Linux CUDA host, install the [compatible vLLM build](docs/installation.md). From the repository root, install the plugin and serve LLaDA2.

```bash
uv pip install -e .
vllm serve inclusionAI/LLaDA2.0-mini \
  --attention-backend TRITON_ATTN --block-size 32 --no-enable-chunked-prefill \
  --max-model-len 2048 --max-num-seqs 8 --max-num-batched-tokens 2048 \
  --generation-config vllm
```

[Send your first request](docs/quickstart.md#send-a-request) · [Serving settings](docs/serving.md) · [Benchmarking](docs/benchmarking.md)

## How it works

![vLLM schedules each denoising step. Both model families share a sampler that holds unfinished blocks and emits tokens from completed blocks.](docs/assets/architecture.svg)
