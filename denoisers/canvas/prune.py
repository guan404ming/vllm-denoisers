import torch

provider = None


def active_rows(
    masked: torch.Tensor, width: torch.Tensor, full: torch.Tensor, shifted: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    block = masked.shape[1]
    if shifted:
        left = torch.zeros_like(masked)
        left[:, :-1] = masked[:, 1:]
        masked = masked | left
    count = masked.sum(dim=1)
    full = full | (count >= width) | (count == 0)
    order = torch.argsort((~masked).to(torch.int8), dim=1, stable=True)
    arange = torch.arange(block, device=masked.device).expand_as(order)
    active = torch.where(full.unsqueeze(1), arange, order)
    length = torch.where(full, torch.full_like(count, block), count)
    return active, length


def inverse_index(
    active: torch.Tensor, widths: torch.Tensor, starts: torch.Tensor
) -> torch.Tensor:
    block = active.shape[1]
    offsets = torch.arange(block, device=active.device)
    src = torch.where(offsets < widths.unsqueeze(1), starts.unsqueeze(1) + offsets, -1)
    return torch.empty_like(active).scatter_(1, active, src)
