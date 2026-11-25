# Prediction feedback

By default, each denoising step reads only the tokens on the canvas. A feedback policy also feeds the predictions of one step into the next step.

## Enable a feedback policy

Set `feedback_policy` in `--hf-overrides`. For a model that supports weighted input embeddings:

```bash
vllm serve <model> --hf-overrides '{"feedback_policy":"smooth"}'
```

| Policy | Behavior |
| --- | --- |
| `smooth` | Add a weighted token embedding at masked positions. |
| `self_conditioning` | Pass the previous soft prediction to the model. |

- The `self_conditioning` policy needs a model with a self-conditioning hook. [DiffusionGemma](../models/diffusion-gemma.md) enables it by default.

[Write a feedback policy](../extending.md#feedback-policy)
