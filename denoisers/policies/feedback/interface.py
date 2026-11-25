from abc import ABC, abstractmethod

import torch

from denoisers.canvas.rows import CanvasRows
from denoisers.utils.registry import Registry


class FeedbackPolicy(ABC):
    hidden_size: int

    @classmethod
    @abstractmethod
    def from_config(cls, config, model) -> "FeedbackPolicy":
        raise NotImplementedError

    @abstractmethod
    def soft(
        self, logits: torch.Tensor, have: torch.Tensor, before: CanvasRows, after: CanvasRows
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def condition(self, embeds: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


registry = Registry("feedback policy", FeedbackPolicy)
