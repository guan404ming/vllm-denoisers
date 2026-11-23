# Extend the plugin

Each policy registry accepts a short name or an importable class path in the server configuration. Register short names from a `vllm.general_plugins` entry point so that server and worker processes can resolve them.

| Policy | Server setting | Existing implementation |
| --- | --- | --- |
| Decode | `decode_policy` | [LowConfidence](https://github.com/guan404ming/vllm-denoisers/blob/main/denoisers/policies/decode/low_confidence.py) |
| KV | `kv_policy` | [Window](https://github.com/guan404ming/vllm-denoisers/blob/main/denoisers/policies/kv/window.py) |
| Input | `input_policy` | [SelfConditioning](https://github.com/guan404ming/vllm-denoisers/blob/main/denoisers/policies/inputs/self_conditioning.py) |
| Hold | `DENOISERS_HOLD` | [ShortestFirst](https://github.com/guan404ming/vllm-denoisers/blob/main/denoisers/policies/hold/sjf.py) |

## Decode policy

Implement `DecodePolicy.from_config(config, model)` and `__call__(logits, rows)`. Return updated `CanvasRows` and a `StepResult`. Set `done` for rows ready to commit. To reuse token sampling and prefix exit, subclass `ScoreMovePolicy` and implement `score` and `move`. A policy that needs previous steps can request a history window with `history_length`.

A request can select a registered decoder by name.

```python
from vllm import SamplingParams

params = SamplingParams(extra_args={"decode_policy": "joint"})
```

The runtime checks `aligned`, `causal_prefill`, `causal_commit`, and `shifted` against the model default. It logs a warning and uses the default if the requested decoder is unknown, fails to build, or requires a different attention layout.

## KV policy

Implement `keep(before, after)` to select positions for the next forward. Return a boolean tensor of shape `[requests, canvas_length]`. Override `prefix` and set `refreshes_prefix = True` to re-encode committed tokens before the canvas. Disable prefix caching for that policy.

The built-in `window` policy uses the decoder's `focus` mask, widened by `prefix_look` and `after_look`. It refreshes the canvas when the focus moves. The `llada` and `fast_dllm` decoders expose sub-block focus. See [KV cache reuse](kv-cache.md) for refresh and async scheduling options.

## Input policy

Implement `soft(logits, have, before, after)` to calculate a soft input and `condition(embeds, soft)` to combine it with token embeddings. The `have` mask identifies positions with logits. See the [smooth policy](https://github.com/guan404ming/vllm-denoisers/blob/main/denoisers/policies/inputs/smooth.py) for a weighted embedding on masked positions and the [DiffusionGemma guide](models/diffusion-gemma.md) for self-conditioning.

## Hold policy

Implement `rank(candidates, now, scores)`. Return request IDs in priority order. Each candidate contains a request ID and an arrival time. The scheduler defers requests beyond the active limit. See [Scheduling](scheduling.md) for a working server configuration.

## Register a model family

For a model that shares the canvas runtime, add the `DecodeConfig` mixin to its Hugging Face config and the `BlockDiffusionModel` mixin to its model class. Call `register_family(architecture, model_path, config_class)` from plugin initialization. Pass a tokenizer or verifier if the checkpoint requires one. Use the [SDAR adapter](https://github.com/guan404ming/vllm-denoisers/blob/main/denoisers/models/sdar/model.py) and [family registrations](https://github.com/guan404ming/vllm-denoisers/blob/main/denoisers/models/families.py) as examples.
