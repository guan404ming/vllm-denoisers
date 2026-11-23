# User guide

## Core Concepts

- [Block decoding](how-it-works.md) · Follow a block of tokens from prompt prefill through denoising to output.
- [Decode policies](decode-policies.md) · Compare the rules for filling masks and revising tokens.

## Serving

- [Batch and canvas limits](serving.md) · Set the batch token budget for the canvas size and number of concurrent requests.
- [Request scheduling](scheduling.md) · Choose which requests advance at each decoding step.
- [KV cache reuse](kv-cache.md) · Choose which token positions to recompute between full passes.
