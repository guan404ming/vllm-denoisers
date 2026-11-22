import torch

from denoisers.ops.stats import row_stats, sample_stats


def given(value, fallback):
    return fallback if value is None else value


def exclude_filtered_mask(
    logits: torch.Tensor, mask_id: int, top_k: torch.Tensor | None, top_p: torch.Tensor | None
) -> None:
    if mask_id < 0:
        return
    mask = logits[:, mask_id]
    if top_k is not None:
        mask.masked_fill_(top_k < logits.shape[-1], -torch.inf)
    if top_p is not None:
        mask.masked_fill_(top_p < 1, -torch.inf)


def uniform(salt: torch.Tensor, step: torch.Tensor, block: int) -> torch.Tensor:
    cells = torch.arange(block, device=salt.device)
    key = (salt.unsqueeze(-1) * 1000003 + step.unsqueeze(-1)) * 1000003 + cells
    key = (key ^ (key >> 30)) * -4658895280553007687
    key = (key ^ (key >> 27)) * -7723592293110705685
    key = key ^ (key >> 31)
    return (key & 0xFFFFFF).float() / (1 << 24)


def draws(rows, temperature: float) -> torch.Tensor | None:
    if temperature <= 0 or rows.salt is None:
        return None
    return uniform(rows.salt, rows.step, rows.canvas.shape[1])


def shift_rows(values: torch.Tensor) -> torch.Tensor:
    return torch.cat([values[:, :1], values[:, :-1]], dim=1)


def transfer_schedule(block_length: int, steps: int) -> torch.Tensor:
    base, remainder = divmod(block_length, steps)
    schedule = torch.full((steps,), base, dtype=torch.int64)
    schedule[:remainder] += 1
    return schedule


def sample(
    logits: torch.Tensor,
    mask_id: int,
    temperature: float = 0.0,
    active: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if temperature <= 0:
        stats = row_stats(logits, mask_id, active=active)
        return stats.x0, stats.p0
    x0, p0, _ = sample_stats(logits, mask_id, temperature, active, noise)
    return x0, p0


def select_unmask(
    masked: torch.Tensor, p0: torch.Tensor, k: torch.Tensor, threshold: float
) -> torch.Tensor:
    confidence = torch.where(masked, p0, -torch.inf)
    k = torch.minimum(k, masked.sum(dim=-1))
    high = confidence > threshold
    rank = confidence.argsort(dim=-1, descending=True, stable=True).argsort(dim=-1)
    top = rank < k.unsqueeze(-1)
    return torch.where((high.sum(dim=-1) >= k).unsqueeze(-1), high, top)


def unmask(
    canvas: torch.Tensor,
    x0: torch.Tensor,
    p0: torch.Tensor,
    k: torch.Tensor,
    threshold: float,
    mask_id: int,
) -> torch.Tensor:
    chosen = select_unmask(canvas == mask_id, p0, k, threshold)
    return torch.where(chosen, x0, canvas)
