from dataclasses import replace

import torch

from denoisers.canvas.rows import CanvasRows, StepResult
from denoisers.policies.decode.interface import DecodePolicy, registry


@registry.register("sigma")
class SigmaLatent(DecodePolicy):
    aligned = False
    causal_commit = True
    commit_logits = False

    def __init__(self, model, calls: int, dim: int):
        self.model = model
        self.calls = calls
        self.dim = dim

    @classmethod
    def from_config(cls, config, model) -> "SigmaLatent":
        return cls(model, config.native_nfe + 1, config.embed_dim)

    def init_canvas(self, canvas: torch.Tensor, slots=None, first: bool = False) -> torch.Tensor:
        return torch.zeros_like(canvas)

    def init_soft(self, soft: torch.Tensor) -> torch.Tensor:
        soft = torch.zeros_like(soft)
        soft[..., : self.dim] = torch.randn_like(soft[..., : self.dim])
        soft[..., -1] = 1
        return soft

    def __call__(self, logits: torch.Tensor, rows: CanvasRows) -> tuple[CanvasRows, StepResult]:
        grid = self.model.gamma_grid(self.calls)
        step = rows.step
        last = step + 1 >= self.calls
        gamma = grid[step.clamp(max=self.calls - 1)].float().view(-1, 1, 1)
        following = grid[(step + 1).clamp(max=self.calls - 1)].float().view(-1, 1, 1)
        matrix = self.model.embedding_matrix()
        z = rows.soft[..., : self.dim]
        prior = torch.sigmoid(-gamma).sqrt() / torch.sigmoid(gamma) * z
        logits = logits.float() + prior @ matrix.t()
        tokens = logits.argmax(-1)
        mean = logits.softmax(-1) @ matrix
        z = torch.sigmoid(-following).sqrt() * mean
        z = z + torch.sigmoid(following).sqrt() * torch.randn_like(mean)
        flag = (step + 2).float().view(-1, 1, 1).expand(*mean.shape[:2], 1)
        soft = torch.where(last.view(-1, 1, 1), 0.0, torch.cat((z, mean, flag), -1))
        canvas = torch.where(last.unsqueeze(1), tokens, rows.canvas)
        return replace(rows, canvas=canvas, soft=soft), StepResult(last)
