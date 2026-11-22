import torch

from denoisers.canvas.rows import CanvasRows
from denoisers.ops.sampling import given, sample, select_unmask
from denoisers.policies.decode.interface import ScoreMovePolicy, registry


@registry.register("fast_dllm")
class FastDLLM(ScoreMovePolicy):
    shifted = True

    def __init__(
        self,
        mask_id: int,
        block_length: int,
        sub_block: int,
        threshold: float,
        temperature: float = 0.0,
    ):
        self.mask_id = mask_id
        self.temperature = temperature
        self.sub_blocks = torch.arange(block_length) // sub_block
        self.threshold = threshold

    @classmethod
    def from_config(cls, config, model) -> "FastDLLM":
        return cls(
            mask_id=config.mask_token_id,
            temperature=config.temperature,
            block_length=config.canvas_length,
            sub_block=config.small_block_size,
            threshold=given(config.confidence_threshold, 0.9),
        )

    def score(
        self,
        logits: torch.Tensor,
        noise: torch.Tensor | None = None,
        active: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return sample(logits, self.mask_id, self.temperature, active, noise)

    def focus(self, rows: CanvasRows) -> torch.Tensor:
        return self._current(rows.canvas)

    def _current(self, canvas: torch.Tensor) -> torch.Tensor:
        masked = canvas == self.mask_id
        self.sub_blocks = sub_blocks = self.sub_blocks.to(masked.device)
        current = torch.where(masked, sub_blocks, sub_blocks.max() + 1).amin(dim=-1)
        return sub_blocks == current.unsqueeze(1)

    def move(self, canvas, step, shift, width, tokens, score):
        active = (canvas == self.mask_id) & self._current(canvas)
        chosen = select_unmask(active, score, torch.ones_like(step), self.threshold) & active
        after = torch.where(chosen, tokens, canvas)
        return after, ~(after == self.mask_id).any(dim=-1)

    def next_seed(self, logits: torch.Tensor) -> torch.Tensor:
        return logits[:, -1].argmax(dim=-1)
