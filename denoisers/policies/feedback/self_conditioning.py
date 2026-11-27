import torch

from denoisers.canvas.rows import CanvasRows
from denoisers.policies.feedback.interface import FeedbackPolicy, registry


@registry.register("self_conditioning")
class SelfConditioning(FeedbackPolicy):
    def __init__(self, model):
        self.model = model
        self.hidden_size = model.input_embedding().weight.shape[1]

    @classmethod
    def from_config(cls, config, model) -> "SelfConditioning":
        return cls(model)

    def soft(
        self, logits: torch.Tensor, have: torch.Tensor, before: CanvasRows, after: CanvasRows
    ) -> torch.Tensor:
        return after.soft * have.unsqueeze(2)

    def condition(self, embeds: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
        return self.model.self_conditioning(embeds, soft)


@registry.register("latent_conditioning")
class LatentConditioning(SelfConditioning):
    def __init__(self, model):
        super().__init__(model)
        self.hidden_size = 2 * model.config.embed_dim + 1
