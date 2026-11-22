# Extending

Three extension points, each an abstract class plus a registry. Pass a registered name or a qualified class name.

| Point | Base class | Registry | Selected by |
|---|---|---|---|
| Decode policy | `denoisers.policies.DecodePolicy` | `register_policy(name)` | `decode_policy` |
| Row pruner | `denoisers.pruners.RowPruner` | `register_pruner(name)` | `pruner` |
| Model family | `denoisers.models.config.DecodeConfig` | `denoisers.plugin.register_family` | the checkpoint architecture |

A qualified class name needs no registration:

```bash
vllm serve JetLM/SDAR-1.7B-Chat --hf-overrides '{"decode_policy": "my_pkg.policy.MyPolicy"}'
```

To use a short name, register it from your own `vllm.general_plugins` entry point, so every engine and worker process sees it.

## Decode policy

Subclass `ScoreMovePolicy` to reuse sampling, shifting, and final prefix exit. Implement `score` (per-cell token and score) and `move` (new canvas and done flags):

```python
from denoisers.policies import ScoreMovePolicy, register_policy


@register_policy("top_k_unmask")
class TopKUnmask(ScoreMovePolicy):
    @classmethod
    def from_config(cls, config, model):
        ...

    def score(self, logits, noise=None, active=None):
        ...

    def move(self, canvas, step, shift, width, tokens, score):
        ...
```

Subclass `DecodePolicy` and implement `__call__(logits, rows)` for a policy that edits the canvas freely. Class attributes declare what the runtime must do: `aligned`, `causal_prefill`, `causal_commit`, `commit_logits`, `sparse_logits`, `shifted`, `stop_exit`, `hidden_size`. Optional hooks: `bind`, `add_request`, `init_canvas`, `next_seed`, `logprob_logits`, `probe_score`.

## Row pruner

A pruner picks the canvas rows that the next denoise step forwards. The other rows keep their cached K/V. `keep(before, after)` returns a `[rows, canvas_length]` bool tensor:

```python
from denoisers.pruners import RowPruner, register_pruner


@register_pruner("low_entropy")
class LowEntropy(RowPruner):
    @classmethod
    def from_config(cls, config):
        ...

    def keep(self, before, after):
        ...
```

The runtime always forwards the full canvas on the first step, the commit step, and steps that need logprobs or probe scores. A pruner turns off async scheduling.

## Model family

Give the HF config the `DecodeConfig` mixin and the model the `BlockDiffusionModel` mixin, then register both:

```python
from denoisers.plugin import register_family

register_family("MyDiffusionLM", "my_pkg.model:MyDiffusionLM", MyDiffusionConfig)
```

Override `DecodeConfig.verify(vllm_config)` for family checks. The shared checks in `denoisers.models.verify` run after it.
