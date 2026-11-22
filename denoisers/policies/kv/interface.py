from abc import ABC, abstractmethod

import torch

from denoisers.canvas.rows import CanvasRows
from denoisers.registry import Registry


class KvPolicy(ABC):
    refreshes_prefix = False

    @classmethod
    @abstractmethod
    def from_config(cls, config, policy) -> "KvPolicy":
        raise NotImplementedError

    @abstractmethod
    def keep(self, before: CanvasRows, after: CanvasRows) -> torch.Tensor:
        raise NotImplementedError

    def prefix(self, before: CanvasRows, after: CanvasRows) -> torch.Tensor:
        return torch.zeros_like(after.step)


registry = Registry("kv policy", KvPolicy)
