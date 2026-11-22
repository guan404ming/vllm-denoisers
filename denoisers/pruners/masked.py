import torch

from denoisers.canvas.transition import CanvasRows
from denoisers.pruners.interface import RowPruner


class MaskedRows(RowPruner):
    def __init__(self, mask_id: int, refresh: int = 0):
        self.mask_id = mask_id
        self.refresh = refresh

    @classmethod
    def from_config(cls, config) -> "MaskedRows":
        return cls(config.mask_token_id, getattr(config, "prune_refresh", 0))

    def keep(self, before: CanvasRows, after: CanvasRows) -> torch.Tensor:
        full = after.committing | (after.step == 0)
        if self.refresh > 0:
            full = full | (after.step % self.refresh == 0)
        return (before.canvas == self.mask_id) | full.unsqueeze(1)
