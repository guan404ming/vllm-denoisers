# Settings reference

This page lists every setting that the plugin reads. Each recipe shows the settings its checkpoint needs.

## Server flags

Pass these flags to `vllm serve`.

| Flag | Use |
| --- | --- |
| `--scheduler-cls` | `aligned`, `elastic`, `resact`, or a class path. Omit it for automatic selection. See [Request scheduling](../features/request-scheduling.md). |
| `--attention-backend` | `TRITON_ATTN` or `FLASH_ATTN`. Omit it for automatic selection. See [Installation](../installation.md#choose-an-attention-backend). |
| `--max-num-batched-tokens` | Batch token budget for one step. See [Batch limits](batch-limits.md). |
| `--max-num-seqs` | Upper limit of concurrent sequences. The plugin lowers it to fit the batch token budget. With `resact`, it sets how many requests stay resident. |
| `--max-model-len` | Longest prompt plus response. |
| `--block-size`, `--prefix-match-unit` | For aligned policies, an explicit value must be a multiple of the canvas length. |
| `--no-enable-prefix-caching` | Required for a KV policy that refreshes the prefix, and for the Dream and LLaDA-8B recipes. |
| `--no-async-scheduling` | Required with a KV policy. |
| `--generation-config vllm` | Use it when checkpoint defaults fail request validation. |
| `--hf-overrides` | Model options, as a JSON object. See the next section. |
| `--diffusion-config` | Engine canvas limit for DiffusionGemma, for example `'{"canvas_length":32}'`. |

## Model options

Pass these keys in `--hf-overrides`. Each recipe lists more keys for its model family.

| Key | Use |
| --- | --- |
| `decode_policy` | Decode policy name. See [Decode policies](../features/decode-policies.md). |
| `kv_policy` | `masked` or `window`. See [KV cache reuse](../features/kv-cache-reuse.md). |
| `kv_refresh` | Step interval for periodic full passes. |
| `prefix_look`, `after_look` | Widen the focus window of the `window` KV policy. |
| `feedback_policy` | `smooth` or `self_conditioning`. See [Prediction feedback](../features/prediction-feedback.md). |
| `lm_head_quantization` | `fp8` or `int4`. See [Output-head quantization](../features/output-head-quantization.md). |
| `canvas_length` | Canvas width, for families that support more than one. |
| `denoise_steps` | Step budget of one canvas. |

## Environment variables

| Variable | Default | Use |
| --- | --- | --- |
| `DENOISERS_HOLD` | `sjf` | Hold policy of the `resact` scheduler: `sjf`, `aging`, `random`, or a class path. |
| `DENOISERS_ACTIVE` | `64` | Resident requests that advance on one step. |
| `DENOISERS_RESERVE` | `0.125` | Share of the active requests reserved for the oldest requests. |
| `DENOISERS_AGE_S` | none | Time scale of the `aging` hold policy, in seconds. |
| `VLLM_PLUGINS` | all plugins | If you set it, include `denoisers`. |

## Request options

Pass these keys in `SamplingParams.extra_args`, or in `vllm_xargs` through the API.

| Key | Use |
| --- | --- |
| `decode_policy` | Select a registered decode policy for one request. See [Extend the plugin](../extending.md#decode-policy). |
| `diffusion_*` | DiffusionGemma request controls. See [DiffusionGemma](../models/diffusion-gemma.md#request-controls). |
