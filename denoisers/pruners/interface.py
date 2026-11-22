from abc import ABC, abstractmethod

import torch

from denoisers.canvas.transition import CanvasRows


class RowPruner(ABC):
    @classmethod
    @abstractmethod
    def from_config(cls, config) -> "RowPruner":
        raise NotImplementedError

    @abstractmethod
    def keep(self, before: CanvasRows, after: CanvasRows) -> torch.Tensor:
        raise NotImplementedError
