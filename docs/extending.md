# Extending

Each policy kind is an abstract class with a `registry`. A config key takes a registered name or a qualified class name.

| Point | Base class | Register with | Selected by |
|---|---|---|---|
| Decode policy | `denoisers.policies.decode.DecodePolicy` | `denoisers.policies.decode.register(name)` | `decode_policy` |
| KV policy | `denoisers.policies.kv.KvPolicy` | `denoisers.policies.kv.register(name)` | `kv_policy` |
| Input policy | `denoisers.policies.inputs.InputPolicy` | `denoisers.policies.inputs.register(name)` | `input_policy` |
| Hold policy | `denoisers.policies.hold.HoldPolicy` | `denoisers.policies.hold.register(name)` | `DENOISERS_HOLD` |
| Model family | `denoisers.models.config.DecodeConfig` | `denoisers.models.families.register_family` | the checkpoint architecture |

A qualified class name needs no registration:

```bash
vllm serve JetLM/SDAR-1.7B-Chat --hf-overrides '{"decode_policy": "my_pkg.policy.MyPolicy"}'
```

To use a short name, register it from your own `vllm.general_plugins` entry point, so every engine and worker process sees it.

## Decode policy

Subclass `ScoreMovePolicy` to reuse sampling, shifting, and final prefix exit. Implement `score` (per-cell token and score) and `move` (new canvas and done flags):

```python
from denoisers.policies.decode import ScoreMovePolicy, register


@register("top_k_unmask")
class TopKUnmask(ScoreMovePolicy):
    @classmethod
    def from_config(cls, config, model): ...

    def score(self, logits, noise=None, active=None): ...

    def move(self, canvas, step, shift, width, tokens, score): ...
```

Subclass `DecodePolicy` and implement `__call__(logits, rows)` for a policy that edits the canvas freely. Return `(rows, StepResult(done, emit, entropy))`. Import `StepResult` from `denoisers.canvas.rows`. Only `done` is required.

Set `history_length` on the decoder to request a history window. The runtime allocates it on first use. Indel defaults to its step limit and accepts a separate `history_length` override. DiffusionGemma uses its stability window.

Class attributes declare what the runtime must do: `aligned`, `causal_prefill`, `causal_commit`, `commit_logits`, `sparse_logits`, `shifted`, `stop_exit`. Optional hooks: `bind`, `add_request`, `init_canvas`, `next_seed`, `focus`, `logprob_logits`, `probe_score`. `focus(rows)` returns the cells the policy can unmask next, or `None` for the whole canvas.

A request can pick a registered decode policy:

```python
SamplingParams(extra_args={"decode_policy": "joint"})
```

Requests take registered names only. A policy that cannot build from the model config, or whose `aligned`, `causal_prefill`, `causal_commit`, or `shifted` differs from the model default, falls back to the default with a warning.

## KV policy

A KV policy picks the canvas cells that the next denoise step forwards. The other cells keep their cached K/V. `keep(before, after)` returns a `[rows, canvas_length]` bool tensor. `prefix(before, after)` returns how many committed tokens before the canvas to re-encode; set `refreshes_prefix = True` when it can be positive.

```python
from denoisers.policies.kv import KvPolicy, register


@register("low_entropy")
class LowEntropy(KvPolicy):
    @classmethod
    def from_config(cls, config, policy): ...

    def keep(self, before, after): ...
```

- `masked`: the cells still masked before the step.
- `window`: the decode policy's `focus`, widened by `prefix_look` committed tokens before it and `after_look` cells after it. It forwards the whole canvas when the focus moves. Use it with policies whose candidates stay inside the focus (`llada`, `fast_dllm`).
- `kv_refresh: N` forwards the whole canvas every N steps.

The runtime always forwards the full canvas on the first step, the commit step, and steps that need logprobs or probe scores. A KV policy turns off async scheduling. A policy that re-encodes the prefix needs `--no-enable-prefix-caching`.

## Input policy

An input policy adds a soft input to the embeddings of the cells that a step forwards. `soft(logits, have, before, after)` returns a `[rows, canvas_length, hidden_size]` tensor from the step's logits; `have` marks the cells that have logits. `condition(embeds, soft)` mixes it into the next step's embeddings.

- `self_conditioning`: DiffusionGemma self-conditioning, the DiffusionGemma default.
- `smooth`: adds `w * softmax(logits) @ E` to still-masked cells, with `w = min(smooth_initial + smooth_growth * step, smooth_weight)` (defaults 0.15, 0.02, 0.3).

## Hold policy

The resident/active scheduler holds every request that ranks past `DENOISERS_ACTIVE`. `rank(candidates, now, scores)` returns request ids, most urgent first. `candidates` holds `(request_id, arrival_time)` pairs and `scores` holds probe scores. `from_env()` builds the policy. Built-ins: `sjf`, `aging` (`DENOISERS_AGE_S`), `random`.

## Model family

Give the HF config the `DecodeConfig` mixin and the model the `BlockDiffusionModel` mixin, then register both:

```python
from denoisers.models.families import register_family

register_family("MyDiffusionLM", "my_pkg.model:MyDiffusionLM", MyDiffusionConfig)
```

Override `DecodeConfig.verify(vllm_config)` for family checks. The shared checks in `denoisers.models.verify` run after it. Override `input_embedding()` on the model when its token embedding is not `model.embed_tokens`.
