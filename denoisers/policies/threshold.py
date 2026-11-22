from dataclasses import replace

import torch

from denoisers.canvas.transition import CanvasRows
from denoisers.policies.sampling import draws, given, sample, transfer_schedule, unmask


class LowConfidence:
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
    def from_config(cls, config) -> "LowConfidence":
        return cls(
            mask_id=config.mask_token_id,
            temperature=config.temperature,
            block_length=config.canvas_length,
            steps=config.denoise_steps,
            threshold=given(config.confidence_threshold, 0.95),
        )

    def __call__(
        self, logits: torch.Tensor, rows: CanvasRows
    ) -> tuple[CanvasRows, torch.Tensor]:
        stats = self.score(logits, draws(rows, self.temperature), rows.canvas == self.mask_id)
        canvas, done = self.move(rows.canvas, rows.step, rows.shift, rows.width, *stats)
        return replace(rows, canvas=canvas), done

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
