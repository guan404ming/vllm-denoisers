# <img src="assets/denoisers.svg" width="40" height="40" alt="" style="vertical-align: text-bottom; margin-inline-end: 0.15em"> vllm-denoisers

Use vLLM to serve models that generate text by refining a block of tokens over several steps.

## Components

[![The scheduler selects requests for model adapters. Token-canvas models use decode policies and optional KV reuse. The shared sampler reports emitted tokens to vLLM.](assets/components.svg)](assets/components.svg)

## How it works

[![Token-canvas and continuous-latent models share a canvas sampler. Unfinished blocks return to the scheduler, and completed blocks emit tokens.](assets/architecture.svg)](assets/architecture.svg)

Read about the [denoising steps](how-it-works.md) or trace a request through the [runtime components](architecture.md).

## Get started

| Task | Start here |
| --- | --- |
| Serve your first model | [Quick start](quickstart.md) |
| Choose a checkpoint and its decoding settings | [Model guide](models.md) |
| Configure serving | [Batch limits](serving.md) · [Request scheduling](scheduling.md) · [KV cache reuse](kv-cache.md) |
| Measure throughput, latency, and accuracy | [Benchmarking](benchmarking.md) |
| Add a policy or model family | [Contributor guide](contributor-guide.md) |
