# Feedback policies

Set `feedback_policy` in `--hf-overrides` to choose a `FeedbackPolicy`. It feeds the predictions of one denoising step into the next step. For a model that supports weighted input embeddings, use `--hf-overrides '{"feedback_policy":"smooth"}'`. The `self_conditioning` policy needs a model with a self-conditioning hook.

| Registered name | Class | Direct base | Behavior |
| --- | --- | --- | --- |
| `smooth` | `Smooth` | `FeedbackPolicy` | Add a weighted token embedding at masked positions. |
| `self_conditioning` | `SelfConditioning` | `FeedbackPolicy` | Pass the previous soft prediction to the model. |

[Feedback policy interface](../extending.md#feedback-policy)
