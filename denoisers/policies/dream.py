from dataclasses import replace

import torch

from denoisers.canvas.transition import CanvasRows
from denoisers.policies.sampling import draws, select_unmask, shift_rows
from denoisers.policies.stats import nucleus_bag, row_stats
from denoisers.scheduler import probe


def transfer_ratios(steps: int, eps: float) -> torch.Tensor:
    timesteps = torch.linspace(1, eps, steps + 1)
    ratios = 1 - timesteps[1:] / timesteps[:-1]
    ratios[-1] = 1
    return ratios


def ratio_table(block_length: int, steps: int, eps: float) -> torch.Tensor:
    table = torch.ones(block_length + 1, steps)
    for masked in range(1, block_length + 1):
        n = max(-(-masked * steps // block_length), 1)
        table[masked, :n] = transfer_ratios(n, eps)
    return table


def nucleus_stats(
    logits: torch.Tensor,
    mask_id: int,
    temperature: float,
    top_p: float,
    active: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    ids, probs = nucleus_bag(logits, mask_id, temperature, active)
    ids, probs = ids.flatten(0, -2), probs.flatten(0, -2)
    probs, order = probs.sort(dim=-1, descending=True)
    q = torch.where(probs.cumsum(dim=-1) - probs <= top_p, probs, 0.0)
    q = q / q.sum(dim=-1, keepdim=True)
    entropy = -(q * torch.log(q + 1e-10)).sum(dim=-1)
    if noise is None:
        noise = torch.rand(q.shape[0], device=q.device)
    below = q.cumsum(dim=-1) < noise.reshape(-1, 1)
    pick = torch.minimum(below.sum(dim=-1), (q > 0).sum(dim=-1) - 1).unsqueeze(-1)
    x0 = ids.gather(-1, order.gather(-1, pick)).squeeze(-1)
    return x0.view(logits.shape[:-1]), entropy.view(logits.shape[:-1])


class Dream:
    shifted = True
    causal_prefill = False
    def __init__(
        self,
        mask_id: int,
        block_length: int,
        steps: int,
        eps: float = 1e-3,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        self.mask_id = mask_id
        self.temperature = temperature
        self.top_p = top_p
        self.table = ratio_table(block_length, steps, eps)
        self.last = (self.table < 1).sum(dim=-1)

    @classmethod
    def from_config(cls, config) -> "Dream":
        return cls(
            mask_id=config.mask_token_id,
            block_length=config.canvas_length,
            steps=config.denoise_steps,
            eps=config.eps,
            temperature=config.temperature,
            top_p=config.top_p,
        )

    def __call__(
        self, logits: torch.Tensor, rows: CanvasRows
    ) -> tuple[CanvasRows, torch.Tensor]:
        masked = rows.canvas == self.mask_id
        need = None
        if not probe.enabled:
            need = torch.cat([masked[:, 1:], torch.zeros_like(masked[:, :1])], dim=1)
            need[:, 0] |= masked[:, 0]
        x0, score = self.score(logits, draws(rows, self.temperature), need)
        x0, score = shift_rows(x0), shift_rows(score)
        canvas, done = self.move(rows.canvas, rows.step, rows.shift, rows.width, x0, score)
        return replace(rows, canvas=canvas, entropy=-score), done

    def score(
        self,
        logits: torch.Tensor,
        noise: torch.Tensor | None = None,
        active: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.temperature > 0:
            x0, entropy = nucleus_stats(
                logits, self.mask_id, self.temperature, self.top_p, active, noise
            )
        else:
            stats = row_stats(logits, self.mask_id, active=active)
            x0, entropy = stats.x0, stats.entropy
        return x0, -entropy

    def move(self, canvas, step, shift, width, tokens, score):
        masked = canvas == self.mask_id
        self.table = table = self.table.to(canvas.device)
        self.last = self.last.to(canvas.device)
        start = (width - shift).clamp(min=0)
        ratio = table[start, step.clamp(max=table.shape[1] - 1)]
        k = (masked.sum(dim=-1).float() * ratio).long()
        chosen = select_unmask(masked, score, k, torch.inf) & masked
        after = torch.where(chosen, tokens, canvas)
        return after, ~(after == self.mask_id).any(dim=-1) | (step >= self.last[start])

    def next_seed(self, logits: torch.Tensor) -> torch.Tensor:
        return logits[:, -1].argmax(dim=-1)
