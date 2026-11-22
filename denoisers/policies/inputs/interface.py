from abc import ABC, abstractmethod

import torch

from denoisers.canvas.rows import CanvasRows
from denoisers.registry import Registry


class InputPolicy(ABC):
    hidden_size: int

    @classmethod
    @abstractmethod
    def from_config(cls, config, model) -> "InputPolicy":
        raise NotImplementedError

    @abstractmethod
    def soft(
        self, logits: torch.Tensor, have: torch.Tensor, before: CanvasRows, after: CanvasRows
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def condition(self, embeds: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


registry = Registry("input policy", InputPolicy)
