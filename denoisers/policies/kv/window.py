import torch

from denoisers.canvas.rows import CanvasRows
from denoisers.policies.kv.interface import KvPolicy, registry


@registry.register("window")
class Window(KvPolicy):
    def __init__(self, policy, prefix_look: int = 0, after_look: int = 0):
        self.policy = policy
        self.prefix_look = prefix_look
        self.after_look = after_look
        self.refreshes_prefix = prefix_look > 0

    @classmethod
    def from_config(cls, config, policy) -> "Window":
        return cls(policy, getattr(config, "prefix_look", 0), getattr(config, "after_look", 0))

    def _bounds(self, rows: CanvasRows) -> tuple[torch.Tensor, torch.Tensor]:
        focus = self.policy.focus(rows)
        n, block = rows.canvas.shape
        if focus is None:
            zero = torch.zeros(n, dtype=torch.int64, device=rows.canvas.device)
            return zero, zero + block
        found = focus.any(dim=1)
        lo = torch.where(found, focus.int().argmax(dim=1), 0)
        hi = torch.where(found, block - focus.flip(1).int().argmax(dim=1), block)
        return lo, hi

    def keep(self, before: CanvasRows, after: CanvasRows) -> torch.Tensor:
        lo, hi = self._bounds(after)
        moved = lo != self._bounds(before)[0]
        cells = torch.arange(after.canvas.shape[1], device=after.canvas.device)
        window = (cells >= (lo - self.prefix_look).unsqueeze(1)) & (
            cells < (hi + self.after_look).unsqueeze(1)
        )
        return window | moved.unsqueeze(1)

    def prefix(self, before: CanvasRows, after: CanvasRows) -> torch.Tensor:
        return (self.prefix_look - self._bounds(after)[0]).clamp(min=0)
