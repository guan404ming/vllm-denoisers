from dataclasses import dataclass

import torch


@dataclass
class CanvasRows:
    canvas: torch.Tensor
    original: torch.Tensor
    step: torch.Tensor
    shift: torch.Tensor
    width: torch.Tensor
    post_steps: torch.Tensor
    committing: torch.Tensor
    history: torch.Tensor | None = None
    history_len: torch.Tensor | None = None
    soft: torch.Tensor | None = None
    slots: torch.Tensor | None = None
    salt: torch.Tensor | None = None
    stops: torch.Tensor | None = None
    start: torch.Tensor | None = None


@dataclass
class StepResult:
    done: torch.Tensor
    emit: torch.Tensor | None = None
    entropy: torch.Tensor | None = None
