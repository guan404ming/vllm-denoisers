import numpy as np
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


class RowPruning:
    def __init__(self, pruner, state, shifted: bool):
        self.pruner = pruner
        self.state = state
        self.shifted = shifted
        rows, block = state.canvas.shape
        self.lengths = torch.full(
            (rows,), block, dtype=torch.int32, pin_memory=torch.cuda.is_available()
        )

    def widths(self, slots_np: np.ndarray) -> np.ndarray:
        return self.lengths.numpy()[slots_np]

    def unpruned(self, slots_np: np.ndarray) -> np.ndarray:
        return self.lengths.numpy()[slots_np] >= self.state.block_length

    def reset(self, slot: int) -> None:
        self.lengths[slot] = self.state.block_length

    def restore(self, slots: torch.Tensor) -> None:
        block = self.state.block_length
        self.state.active[slots] = torch.arange(block, device=slots.device)
        self.state.active_len[slots] = block

    def update(self, slots: torch.Tensor, before, after, full: bool) -> torch.Tensor:
        cells = torch.arange(self.state.block_length, device=slots.device)
        keep = self.pruner.keep(before, after) & (cells < before.width.unsqueeze(1))
        forced = torch.full_like(after.committing, full)
        active, length = active_rows(keep, before.width, forced, self.shifted)
        self.state.active[slots] = active
        self.state.active_len[slots] = length.to(torch.int32)
        return active

    def sync(self) -> None:
        self.lengths.copy_(self.state.active_len, non_blocking=True)

    def pruned(self, slots: torch.Tensor, num_logits: torch.Tensor) -> torch.Tensor:
        return (self.state.active_len[slots] < self.state.block_length) & (num_logits > 0)

    def offsets(
        self, slots: torch.Tensor, rows: torch.Tensor, local: torch.Tensor, num_logits: torch.Tensor
    ) -> torch.Tensor:
        active = self.state.active[slots][rows, local.clamp(max=self.state.block_length - 1)]
        return torch.where(self.pruned(slots, num_logits)[rows], active - local, 0)
