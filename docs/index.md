# <img src="assets/denoisers.svg" width="48" height="48" alt="" style="vertical-align: calc(0.5cap - 24px); margin-inline-end: 0.08em"> vllm-denoisers

Use vLLM to serve models that generate text by refining a block of tokens over several steps.

## Components

[![The scheduler selects requests for model adapters. Token-canvas models use decode policies and optional KV reuse. The shared sampler reports emitted tokens to vLLM.](assets/components.svg)](assets/components.svg)

Browse the features in the [user guide](user-guide.md).

## How it works

[![Token-canvas and continuous-latent models share a canvas sampler. Unfinished blocks return to the scheduler, and completed blocks emit tokens.](assets/architecture.svg)](assets/architecture.svg)

Read about the [denoising steps](concepts/block-decoding.md) or trace a request through the [runtime components](architecture.md).

## Get started

| Task | Start here |
| --- | --- |
| Serve your first model | [Quick start](quickstart.md) |
| Choose a checkpoint and its decoding settings | [Recipes](models.md) |
| Configure serving | [Batch limits](configuration/batch-limits.md) · [Request scheduling](features/request-scheduling.md) · [KV cache reuse](features/kv-cache-reuse.md) |
| Measure throughput, latency, and accuracy | [Benchmarking](benchmarking.md) |
| Add a policy or model family | [Developer guide](contributor-guide.md) |
