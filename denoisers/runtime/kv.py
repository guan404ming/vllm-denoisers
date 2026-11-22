import numpy as np
import torch

from denoisers.canvas.rows import CanvasRows


def layout(
    keep: torch.Tensor,
    width: torch.Tensor,
    full: torch.Tensor,
    shifted: bool,
    prefix: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    block = keep.shape[1]
    if shifted:
        left = torch.zeros_like(keep)
        left[:, :-1] = keep[:, 1:]
        keep = keep | left
    count = keep.sum(dim=1)
    full = full | (count >= width) | (count == 0)
    if prefix is None:
        prefix = torch.zeros_like(count)
    prefix = torch.where(full, 0, torch.minimum(prefix.clamp(min=0), width - count))
    cells = torch.arange(block, device=keep.device).expand_as(keep)
    order = torch.argsort((~keep).to(torch.int8), dim=1, stable=True)
    lead = prefix.unsqueeze(1)
    moved = order.gather(1, (cells - lead).clamp(min=0))
    active = torch.where(cells < lead, cells - lead, moved)
    active = torch.where(full.unsqueeze(1), cells, active)
    length = torch.where(full, torch.full_like(count, block), count + prefix)
    return active, length


def inverse_index(active: torch.Tensor, widths: torch.Tensor, starts: torch.Tensor) -> torch.Tensor:
    n, block = active.shape
    offsets = torch.arange(block, device=active.device)
    valid = (offsets < widths.unsqueeze(1)) & (active >= 0)
    src = torch.where(valid, starts.unsqueeze(1) + offsets, -1)
    index = torch.where(valid, active, block)
    inv = torch.full((n, block + 1), -1, dtype=active.dtype, device=active.device)
    return inv.scatter_(1, index, src)[:, :block].contiguous()


class KvRuntime:
    def __init__(self, build_policy, state, shifted: bool, refresh_every: int = 0):
        self.build_policy = build_policy
        self.policies = {}
        self.state = state
        self.shifted = shifted
        self.refresh_every = refresh_every
        rows, block = state.canvas.shape
        pinned = torch.cuda.is_available()
        self.lengths = torch.full((rows,), block, dtype=torch.int32, pin_memory=pinned)
        self.leads = torch.zeros(rows, dtype=torch.int64, pin_memory=pinned)

    def for_decoder(self, decoder):
        key = id(decoder)
        if key not in self.policies:
            self.policies[key] = self.build_policy(decoder)
        return self.policies[key]

    def widths(self, slots_np: np.ndarray) -> np.ndarray:
        return self.lengths.numpy()[slots_np]

    def identity(self, slots_np: np.ndarray) -> np.ndarray:
        block = self.state.block_length
        return (self.lengths.numpy()[slots_np] >= block) & (self.leads.numpy()[slots_np] >= 0)

    def reset(self, slot: int) -> None:
        self.lengths[slot] = self.state.block_length
        self.leads[slot] = 0

    def restore(self, slots: torch.Tensor) -> None:
        block = self.state.block_length
        self.state.active[slots] = torch.arange(block, device=slots.device)
        self.state.active_len[slots] = block

    def update(
        self, policy, slots: torch.Tensor, before: CanvasRows, after: CanvasRows, full: bool
    ) -> torch.Tensor:
        cells = torch.arange(self.state.block_length, device=slots.device)
        keep = policy.keep(before, after) & (cells < before.width.unsqueeze(1))
        forced = after.committing | (after.step == 0)
        if self.refresh_every > 0:
            forced = forced | (after.step % self.refresh_every == 0)
        if full:
            forced = torch.ones_like(forced)
        prefix = torch.minimum(policy.prefix(before, after), after.start)
        active, length = layout(keep, before.width, forced, self.shifted, prefix)
        self.state.active[slots] = active
        self.state.active_len[slots] = length.to(torch.int32)
        return active

    def draft_tokens(
        self, policy, after: CanvasRows, active: torch.Tensor, history: torch.Tensor
    ) -> torch.Tensor:
        tokens = after.canvas.gather(1, active.clamp(min=0))
        if not policy.refreshes_prefix:
            return tokens
        positions = (after.start.unsqueeze(1) + active).clamp(0, history.shape[1] - 1)
        past = history[after.slots.unsqueeze(1), positions].to(tokens.dtype)
        return torch.where(active < 0, past, tokens)

    def sync(self) -> None:
        self.lengths.copy_(self.state.active_len, non_blocking=True)
        self.leads.copy_(self.state.active[:, 0], non_blocking=True)

    def pruned(self, slots: torch.Tensor, num_logits: torch.Tensor) -> torch.Tensor:
        block = self.state.block_length
        reordered = (self.state.active_len[slots] < block) | (self.state.active[slots, 0] < 0)
        return reordered & (num_logits > 0)

    def offsets(
        self, slots: torch.Tensor, rows: torch.Tensor, local: torch.Tensor, num_logits: torch.Tensor
    ) -> torch.Tensor:
        active = self.state.active[slots][rows, local.clamp(max=self.state.block_length - 1)]
        return torch.where(self.pruned(slots, num_logits)[rows], active - local, 0)
