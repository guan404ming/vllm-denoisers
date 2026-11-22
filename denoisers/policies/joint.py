from dataclasses import replace

import torch

from denoisers.canvas.transition import CanvasRows
from denoisers.policies.interface import DecodePolicy
from denoisers.policies.sampling import given, sample, unmask


class JointThreshold(DecodePolicy):
    commit_logits = False

    def __init__(
        self,
        mask_id: int,
        threshold: float,
        editing_threshold: float,
        max_post_steps: int,
        temperature: float = 0.0,
    ):
        self.mask_id = mask_id
        self.temperature = temperature
        self.threshold = threshold
        self.editing_threshold = editing_threshold
        self.max_post_steps = max_post_steps

    @classmethod
    def from_config(cls, config, model) -> "JointThreshold":
        return cls(
            mask_id=config.mask_token_id,
            temperature=config.temperature,
            threshold=given(config.confidence_threshold, 0.95),
            editing_threshold=given(config.editing_threshold, 0.9),
            max_post_steps=config.max_post_steps,
        )

    def __call__(
        self, logits: torch.Tensor, rows: CanvasRows
    ) -> tuple[CanvasRows, torch.Tensor]:
        canvas = rows.canvas
        x0, p0 = sample(logits, self.mask_id, self.temperature)
        had_masks = (canvas == self.mask_id).any(dim=-1)
        k = torch.ones_like(rows.step)
        unmasked = unmask(canvas, x0, p0, k, self.threshold, self.mask_id)
        positions = torch.arange(canvas.shape[1], device=canvas.device)
        editable = (canvas != self.mask_id) & (positions >= rows.shift.unsqueeze(1))
        edit = editable & (p0 > self.editing_threshold) & (x0 != canvas)
        settled = ~edit.any(dim=-1) | (rows.post_steps + 1 >= self.max_post_steps)
        after = replace(
            rows,
            canvas=torch.where(edit, x0, unmasked),
            post_steps=rows.post_steps + (~had_masks).long(),
        )
        return after, ~had_masks & settled
