from dataclasses import replace

import torch

from denoisers.canvas.transition import CanvasRows
from denoisers.policies.sampling import draws, given, sample, select_unmask, transfer_schedule
from denoisers.policies.stats import sample_stats


class LLaDA:
    causal_prefill = False

    def __init__(
        self,
        mask_id: int,
        block_length: int,
        sub_block: int,
        steps: int,
        temperature: float = 0.0,
    ):
        self.mask_id = mask_id
        self.temperature = temperature
        self.sub_block = sub_block
        self.steps_per_block = max(steps // (block_length // sub_block), 1)
        self.schedule = transfer_schedule(sub_block, self.steps_per_block)
        self.last = self.steps_per_block * (block_length // sub_block) - 1

    @classmethod
    def from_config(cls, config) -> "LLaDA":
        return cls(
            mask_id=config.mask_token_id,
            block_length=config.canvas_length,
            sub_block=config.sub_block_length,
            steps=given(config.denoise_steps, config.canvas_length),
            temperature=config.temperature,
        )

    def __call__(
        self, logits: torch.Tensor, rows: CanvasRows
    ) -> tuple[CanvasRows, torch.Tensor]:
        stats = self.score(
            logits, draws(rows, self.temperature), self._active(rows.canvas, rows.shift)
        )
        canvas, done = self.move(rows.canvas, rows.step, rows.shift, rows.width, *stats)
        return replace(rows, canvas=canvas), done

    def _active(self, canvas: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
        masked = canvas == self.mask_id
        offset = torch.arange(masked.shape[1], device=masked.device) - shift.unsqueeze(1)
        sub_blocks = offset.div(self.sub_block, rounding_mode="floor")
        current = torch.where(masked, sub_blocks, masked.shape[1]).amin(dim=-1)
        return masked & (sub_blocks == current.unsqueeze(1))

    def score(
        self,
        logits: torch.Tensor,
        noise: torch.Tensor | None = None,
        active: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.temperature <= 0:
            return sample(logits, self.mask_id, 0.0, active)
        x0, _, p0 = sample_stats(logits, self.mask_id, self.temperature, active, noise)
        return x0, p0

    def move(self, canvas, step, shift, width, tokens, score):
        active = self._active(canvas, shift)
        self.schedule = self.schedule.to(canvas.device)
        k = self.schedule[step % self.steps_per_block]
        chosen = select_unmask(active, score, k, torch.inf) & active
        after = torch.where(chosen, tokens, canvas)
        return after, ~(after == self.mask_id).any(dim=-1) | (step >= self.last)
