from abc import ABC, abstractmethod
from dataclasses import replace

import torch

from denoisers.canvas.rows import CanvasRows
from denoisers.ops.sampling import draws, shift_rows
from denoisers.registry import Registry
from denoisers.scheduler.probe import mean_entropy


class DecodePolicy(ABC):
    aligned = True
    causal_prefill = True
    causal_commit = False
    commit_logits = True
    sparse_logits = False
    shifted = False
    stop_exit = False
    dense_scores = False

    @classmethod
    @abstractmethod
    def from_config(cls, config, model) -> "DecodePolicy":
        raise NotImplementedError

    @abstractmethod
    def __call__(self, logits: torch.Tensor, rows: CanvasRows) -> tuple[CanvasRows, torch.Tensor]:
        raise NotImplementedError

    def bind(self, state) -> None:
        pass

    def add_request(self, slot: int, sampling_params) -> None:
        pass

    def init_canvas(self, canvas: torch.Tensor, slots=None, first: bool = False) -> torch.Tensor:
        return canvas

    def next_seed(self, logits: torch.Tensor) -> torch.Tensor | None:
        return None

    def focus(self, rows: CanvasRows) -> torch.Tensor | None:
        return None

    def logprob_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return logits

    def probe_score(
        self, entropy: torch.Tensor, before: CanvasRows, after: CanvasRows
    ) -> torch.Tensor:
        cells = torch.arange(entropy.shape[1], device=entropy.device)
        return mean_entropy(entropy, cells >= before.shift.unsqueeze(1))


class ScoreMovePolicy(DecodePolicy):
    stop_exit = True
    mask_id: int
    temperature: float

    def active(self, rows: CanvasRows) -> torch.Tensor | None:
        return None

    @abstractmethod
    def score(
        self,
        logits: torch.Tensor,
        noise: torch.Tensor | None = None,
        active: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def move(self, canvas, step, shift, width, tokens, score) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    def __call__(self, logits: torch.Tensor, rows: CanvasRows) -> tuple[CanvasRows, torch.Tensor]:
        tokens, score = self.score(logits, draws(rows, self.temperature), self.active(rows))
        if self.shifted:
            tokens, score = shift_rows(tokens), shift_rows(score)
        canvas, done = self.move(rows.canvas, rows.step, rows.shift, rows.width, tokens, score)
        return replace(rows, canvas=canvas), done


registry = Registry("decode policy", DecodePolicy)
