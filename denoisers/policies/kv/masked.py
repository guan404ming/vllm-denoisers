import torch

from denoisers.canvas.rows import CanvasRows
from denoisers.policies.kv.interface import KvPolicy, registry


@registry.register("masked")
class Masked(KvPolicy):
    def __init__(self, mask_id: int):
        self.mask_id = mask_id

    @classmethod
    def from_config(cls, config, policy) -> "Masked":
        return cls(config.mask_token_id)

    def keep(self, before: CanvasRows, after: CanvasRows) -> torch.Tensor:
        return before.canvas == self.mask_id
