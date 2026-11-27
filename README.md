<h1 align="center">
<sup><img src="https://raw.githubusercontent.com/guan404ming/vllm-denoisers/main/docs/assets/denoisers.svg" width="52" height="52" align="middle" alt=""></sup> vllm-denoisers
</h1>

<h3 align="center">
Serve diffusion language models through vLLM
</h3>

<p align="center">
| <a href="https://guan404ming.github.io/vllm-denoisers/"><b>Documentation</b></a> | <a href="https://guan404ming.github.io/vllm-denoisers/quickstart/"><b>Quick Start</b></a> | <a href="https://guan404ming.github.io/vllm-denoisers/models/"><b>Recipes</b></a> | <a href="https://guan404ming.github.io/vllm-denoisers/user-guide/"><b>User Guide</b></a> | <a href="https://guan404ming.github.io/vllm-denoisers/contributor-guide/"><b>Developer Guide</b></a> |
</p>

<p align="center">
<a href="https://github.com/guan404ming/vllm-denoisers/actions/workflows/docs.yml"><img alt="Documentation CI" src="https://github.com/guan404ming/vllm-denoisers/actions/workflows/docs.yml/badge.svg?branch=main"></a>
<a href="https://pypi.org/project/vllm-denoisers/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/vllm-denoisers.svg"></a>
<a href="https://github.com/guan404ming/vllm-denoisers/blob/main/pyproject.toml"><img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-blue.svg"></a>
<a href="https://github.com/vllm-project/vllm/releases/tag/v0.34.0"><img alt="vLLM 0.34" src="https://img.shields.io/badge/vLLM-0.34-orange.svg"></a>
</p>

---

## About

vllm-denoisers is a vLLM plugin for serving diffusion language models.

Each request refines a block of tokens over several steps, then emits the completed tokens through vLLM’s OpenAI-compatible API.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/guan404ming/vllm-denoisers/main/docs/assets/architecture.svg">
    <img src="https://raw.githubusercontent.com/guan404ming/vllm-denoisers/main/docs/assets/architecture-light.svg" width="75%" alt="vLLM schedules each denoising step. Both model families share a sampler that holds unfinished blocks and emits tokens from completed blocks.">
  </picture>
</p>

vllm-denoisers is efficient with:

- Denoising steps batched and scheduled by vLLM
- KV cache reuse between full passes
- A specialized FlashAttention 4 backend for block-aligned masking
- Resident/active scheduling by expected remaining steps
- FP8 and INT4 output-head quantization
- Tensor parallelism for token-canvas models

vllm-denoisers is flexible and easy to use with:

- Installation as a vLLM plugin, served with `vllm serve`
- Replaceable decode, KV, feedback, and scheduling policies
- Registries for custom policies and model families

vllm-denoisers supports these model families:

| Model type | Families |
| --- | --- |
| Token-canvas | [DiffuCoder](https://guan404ming.github.io/vllm-denoisers/models/diffucoder/), [DiffusionGemma](https://guan404ming.github.io/vllm-denoisers/models/diffusion-gemma/), [dLLM BD3LM](https://guan404ming.github.io/vllm-denoisers/models/dllm-bd3lm/), [Dream](https://guan404ming.github.io/vllm-denoisers/models/dream/), [Fast-dLLM v2](https://guan404ming.github.io/vllm-denoisers/models/fast-dllm-v2/), [LLaDA-8B](https://guan404ming.github.io/vllm-denoisers/models/llada/), [LLaDA2](https://guan404ming.github.io/vllm-denoisers/models/llada2/), [LLaDA-MoE](https://guan404ming.github.io/vllm-denoisers/models/llada-moe/), [Nemotron Labs Diffusion](https://guan404ming.github.io/vllm-denoisers/models/nemotron-labs-diffusion/), [SDAR](https://guan404ming.github.io/vllm-denoisers/models/sdar-trado/), [SDLLM-MDLM](https://guan404ming.github.io/vllm-denoisers/models/sdllm/), [Stable-DiffCoder](https://guan404ming.github.io/vllm-denoisers/models/stable-diffcoder/), [TraDo](https://guan404ming.github.io/vllm-denoisers/models/trado/) |
| Continuous-latent | [Nemotron Labs Sigma](https://guan404ming.github.io/vllm-denoisers/models/nemotron-labs-sigma/) |

## Getting Started

On Linux with Python 3.12 or later and a GPU that vLLM supports, install vLLM 0.34 and the plugin:

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install 'vllm-denoisers[vllm]'
```

Serve LLaDA2:

```bash
vllm serve inclusionAI/LLaDA2.0-mini --max-model-len 2048 --generation-config vllm
```

Visit our [documentation](https://guan404ming.github.io/vllm-denoisers/) to learn more.

- [Installation](https://guan404ming.github.io/vllm-denoisers/installation/)
- [Quick Start](https://guan404ming.github.io/vllm-denoisers/quickstart/)
- [Settings Reference](https://guan404ming.github.io/vllm-denoisers/configuration/settings/)
- [Benchmarking](https://guan404ming.github.io/vllm-denoisers/benchmarking/)

## Versioning

Follows [Semantic Versioning](https://semver.org/) with upstream tracking:

- MAJOR.MINOR: Tracks upstream [vLLM](https://github.com/vllm-project/vllm)
- PATCH: vllm-denoisers specific releases

Example: `0.34.2` = vLLM 0.34 + 2 vllm-denoisers patch releases

## Contributing

We welcome and value any contributions and collaborations.
Please check out the [Developer Guide](https://guan404ming.github.io/vllm-denoisers/contributor-guide/) for the runtime architecture, extension points, and testing.

## Contact Us

For technical questions and feature requests, please use GitHub [Issues](https://github.com/guan404ming/vllm-denoisers/issues).
