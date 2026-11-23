# <sub><img src="docs/assets/denoisers.svg" width="44" height="44" alt=""></sub>&#x2004;vllm-denoisers

![vLLM schedules each denoising step. Both model families share a sampler that holds unfinished blocks and emits tokens from completed blocks.](docs/assets/architecture.svg)

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)

Serve diffusion language models through vLLM’s OpenAI-compatible API. Each request refines a block over several steps, then emits the completed tokens.

Supported families include LLaDA2, Dream, DiffusionGemma, and Nemotron Labs Sigma.

[Documentation](https://guan404ming.github.io/vllm-denoisers/) · [Full model list](https://guan404ming.github.io/vllm-denoisers/models/)

## Quick start

On a Linux CUDA host, install the [compatible vLLM build](https://guan404ming.github.io/vllm-denoisers/installation/). From the repository root, install the plugin and serve LLaDA2.

```bash
uv pip install -e .
vllm serve inclusionAI/LLaDA2.0-mini \
  --attention-backend TRITON_ATTN --block-size 32 \
  --max-model-len 2048 --max-num-seqs 8 --max-num-batched-tokens 2048 \
  --generation-config vllm
```

[Send your first request](https://guan404ming.github.io/vllm-denoisers/quickstart/#send-a-request) · [Serving settings](https://guan404ming.github.io/vllm-denoisers/serving/) · [Benchmarking](https://guan404ming.github.io/vllm-denoisers/benchmarking/)
