<h1 align="center">
<sup><img src="docs/assets/denoisers.svg" width="52" height="52" align="middle" alt=""></sup> vllm-denoisers
</h1>

<h3 align="center">
Serve diffusion language models through vLLM
</h3>

<p align="center">
| <a href="https://guan404ming.github.io/vllm-denoisers/"><b>Documentation</b></a> | <a href="https://guan404ming.github.io/vllm-denoisers/quickstart/"><b>Quick Start</b></a> | <a href="https://guan404ming.github.io/vllm-denoisers/models/"><b>Recipes</b></a> | <a href="https://guan404ming.github.io/vllm-denoisers/components/"><b>Components</b></a> | <a href="https://guan404ming.github.io/vllm-denoisers/contributor-guide/"><b>Developer Guide</b></a> |
</p>

<p align="center">
<a href="https://github.com/guan404ming/vllm-denoisers/actions/workflows/docs.yml"><img alt="Documentation CI" src="https://github.com/guan404ming/vllm-denoisers/actions/workflows/docs.yml/badge.svg?branch=main"></a>
<a href="pyproject.toml"><img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-blue.svg"></a>
</p>

---

## About

vllm-denoisers is a vLLM plugin for serving diffusion language models.

Each request refines a block of tokens over several steps, then emits the completed tokens through vLLM’s OpenAI-compatible API.

<p align="center">
  <img src="docs/assets/architecture.svg" width="75%" alt="vLLM schedules each denoising step. Both model families share a sampler that holds unfinished blocks and emits tokens from completed blocks.">
</p>

vllm-denoisers is efficient with:

- Denoising steps batched and scheduled by vLLM
- KV cache reuse between full passes
- Resident/active scheduling by expected remaining steps
- FP8 and INT4 output-head quantization
- Tensor parallelism for token-canvas models

vllm-denoisers is flexible and easy to use with:

- Installation as a vLLM plugin, served with `vllm serve`
- Replaceable decode, KV, input, and scheduling policies
- Registries for custom policies and model families

vllm-denoisers supports these model families:

| Model type | Families |
| --- | --- |
| Token-canvas | [DiffusionGemma](https://guan404ming.github.io/vllm-denoisers/models/diffusion-gemma/), [Dream](https://guan404ming.github.io/vllm-denoisers/models/dream/), [Fast-dLLM v2](https://guan404ming.github.io/vllm-denoisers/models/fast-dllm-v2/), [LLaDA-8B](https://guan404ming.github.io/vllm-denoisers/models/llada/), [LLaDA2](https://guan404ming.github.io/vllm-denoisers/models/llada2/), [Nemotron Labs Diffusion](https://guan404ming.github.io/vllm-denoisers/models/nemotron-labs-diffusion/), [SDAR](https://guan404ming.github.io/vllm-denoisers/models/sdar-trado/), [Stable-DiffCoder](https://guan404ming.github.io/vllm-denoisers/models/stable-diffcoder/), [TraDo](https://guan404ming.github.io/vllm-denoisers/models/trado/) |
| Continuous-latent | [Nemotron Labs Sigma](https://guan404ming.github.io/vllm-denoisers/models/nemotron-labs-sigma/) |

## Getting Started

On Linux with Python 3.12 or later and a CUDA GPU, install the compatible vLLM build and the plugin from the repository root:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install 'vllm==0.33.0rc1.dev487+g5689c2790' \
  --extra-index-url https://wheels.vllm.ai/5689c2790c911a55c75825aac5166305df0b591d/
uv pip install -e .
```

Serve LLaDA2:

```bash
vllm serve inclusionAI/LLaDA2.0-mini \
  --attention-backend TRITON_ATTN --block-size 32 \
  --max-model-len 2048 --max-num-seqs 8 --max-num-batched-tokens 2048 \
  --generation-config vllm
```

Visit our [documentation](https://guan404ming.github.io/vllm-denoisers/) to learn more.

- [Installation](https://guan404ming.github.io/vllm-denoisers/installation/)
- [Quick Start](https://guan404ming.github.io/vllm-denoisers/quickstart/)
- [Serving Settings](https://guan404ming.github.io/vllm-denoisers/serving/)
- [Benchmarking](https://guan404ming.github.io/vllm-denoisers/benchmarking/)

## Contributing

We welcome and value any contributions and collaborations.
Please check out the [Developer Guide](https://guan404ming.github.io/vllm-denoisers/contributor-guide/) for the runtime architecture, extension points, and testing.

## Contact Us

For technical questions and feature requests, please use GitHub [Issues](https://github.com/guan404ming/vllm-denoisers/issues).
