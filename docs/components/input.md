# Input policies

Set `input_policy` in `--hf-overrides` to choose an `InputPolicy` for the next denoising step. For a model that supports weighted input embeddings, use `--hf-overrides '{"input_policy":"smooth"}'`. The `self_conditioning` policy needs a model with a self-conditioning hook.

| Registered name | Class | Direct base | Behavior |
| --- | --- | --- | --- |
| `smooth` | `Smooth` | `InputPolicy` | Add a weighted token embedding at masked positions. |
| `self_conditioning` | `SelfConditioning` | `InputPolicy` | Pass the previous soft prediction to the model. |

[Input policy interface](../extending.md#input-policy)
