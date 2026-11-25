# User guide

## Concepts

- [How block decoding works](concepts/block-decoding.md) · Follow a block of tokens from prompt prefill through denoising to output.
- [Denoising step](concepts/denoising-step.md) · See what happens in one step and which feature controls each stage.

## Features

- [Decode policies](features/decode-policies.md) · Compare the rules for filling masks and revising tokens.
- [KV cache reuse](features/kv-cache-reuse.md) · Choose which token positions to recompute between full passes.
- [Prediction feedback](features/prediction-feedback.md) · Feed the predictions of one denoising step into the next step.
- [Request scheduling](features/request-scheduling.md) · Choose which requests advance at each decoding step.
- [Output-head quantization](features/output-head-quantization.md) · Quantize the output head to FP8 or INT4.

## Configuration

- [Batch limits](configuration/batch-limits.md) · Set the batch token budget for the canvas size and number of concurrent requests.
- [Settings reference](configuration/settings.md) · Find every server flag, model option, environment variable, and request option.
