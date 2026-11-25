# Denoising step

A block needs several denoising steps, and each step is one vLLM batch. This page follows a token-canvas model through one step and shows which feature controls each stage.

[![The scheduler selects requests for model adapters. Token-canvas models use decode policies and optional KV reuse. The shared sampler reports emitted tokens to vLLM.](../assets/components.svg){ width="100%" }](../assets/components.svg)

| Stage | What happens | Feature |
| --- | --- | --- |
| 1.&nbsp;Schedule | The scheduler selects the requests that advance on this step. | [Request scheduling](../features/request-scheduling.md) |
| 2.&nbsp;Forward | The model reads the canvas of each selected request. With a KV policy, it recomputes only the positions selected after the previous step. | [KV cache reuse](../features/kv-cache-reuse.md) |
| 3.&nbsp;Score | The output head scores candidate tokens at the canvas positions. | [Output-head quantization](../features/output-head-quantization.md) |
| 4.&nbsp;Update | The decode policy uses the scores to update the canvas and signals when the block is ready to commit. | [Decode policies](../features/decode-policies.md) |
| 5.&nbsp;Feedback | A feedback policy can keep the predictions of this step for the next step. | [Prediction feedback](../features/prediction-feedback.md) |
| 6.&nbsp;Report | The sampler reports zero output tokens while the block is in progress. At commit, it reports the tokens of the block. | |

[Block decoding](block-decoding.md) · [Runtime architecture](../architecture.md)
