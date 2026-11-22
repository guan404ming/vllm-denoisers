import torch

from denoisers.canvas.rows import CanvasRows
from denoisers.ops.sampling import given, sample, transfer_schedule, unmask
from denoisers.policies.decode.interface import ScoreMovePolicy, registry


@registry.register("low_confidence")
class LowConfidence(ScoreMovePolicy):
    sparse_logits = True

    def __init__(
        self,
        mask_id: int,
        block_length: int,
        steps: int,
        threshold: float,
        temperature: float = 0.0,
    ):
        self.mask_id = mask_id
        self.temperature = temperature
        self.threshold = threshold
        self.schedule = transfer_schedule(block_length, steps)

    @classmethod
    def from_config(cls, config, model) -> "LowConfidence":
        return cls(
            mask_id=config.mask_token_id,
            temperature=config.temperature,
            block_length=config.canvas_length,
            steps=config.denoise_steps,
            threshold=given(config.confidence_threshold, 0.95),
        )

    def active(self, rows: CanvasRows) -> torch.Tensor:
        return rows.canvas == self.mask_id

    def score(
        self,
        logits: torch.Tensor,
        noise: torch.Tensor | None = None,
        active: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return sample(logits, self.mask_id, self.temperature, active, noise)

    def move(self, canvas, step, shift, width, tokens, score):
        self.schedule = self.schedule.to(canvas.device)
        last = len(self.schedule) - 1
        k = self.schedule[step.clamp(max=last)]
        after = unmask(canvas, tokens, score, k, self.threshold, self.mask_id)
        return after, ~(after == self.mask_id).any(dim=-1) | (step >= last)
