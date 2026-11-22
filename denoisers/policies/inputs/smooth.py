import torch

from denoisers.canvas.rows import CanvasRows
from denoisers.ops.embedding import ExpectedEmbedding
from denoisers.policies.inputs.interface import InputPolicy, registry

try:
    from vllm.distributed import get_tensor_model_parallel_world_size
except ImportError:
    get_tensor_model_parallel_world_size = None


@registry.register("smooth")
class Smooth(InputPolicy):
    def __init__(
        self,
        embedding: ExpectedEmbedding,
        mask_id: int,
        weight: float = 0.3,
        initial: float = 0.15,
        growth: float = 0.02,
    ):
        self.embedding = embedding
        self.mask_id = mask_id
        self.weight = weight
        self.initial = initial
        self.growth = growth
        self.hidden_size = embedding.hidden_size

    @classmethod
    def from_config(cls, config, model) -> "Smooth":
        return cls(
            ExpectedEmbedding.of(model.input_embedding(), get_tensor_model_parallel_world_size()),
            config.mask_token_id,
            getattr(config, "smooth_weight", 0.3),
            getattr(config, "smooth_initial", 0.15),
            getattr(config, "smooth_growth", 0.02),
        )

    def soft(
        self, logits: torch.Tensor, have: torch.Tensor, before: CanvasRows, after: CanvasRows
    ) -> torch.Tensor:
        n, block, vocab = logits.shape
        cells = (after.canvas == self.mask_id) & have & ~before.committing.unsqueeze(1)
        scale = (self.initial + self.growth * before.step.float()).clamp(max=self.weight)
        out = torch.zeros(n, block, self.hidden_size, dtype=torch.float32, device=logits.device)
        index = cells.view(-1).nonzero().squeeze(1)
        probs = torch.softmax(logits.view(-1, vocab)[index].float(), dim=-1)
        rows = index // block
        out.view(-1, self.hidden_size)[index] = self.embedding(probs).float() * scale[rows, None]
        return out

    def condition(self, embeds: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
        return embeds + soft
