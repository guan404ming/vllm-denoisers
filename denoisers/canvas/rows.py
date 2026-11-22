from dataclasses import dataclass, fields, replace

import torch


@dataclass
class CanvasRows:
    canvas: torch.Tensor
    original: torch.Tensor
    step: torch.Tensor
    shift: torch.Tensor
    width: torch.Tensor
    post_steps: torch.Tensor
    history: torch.Tensor
    history_len: torch.Tensor
    committing: torch.Tensor
    soft: torch.Tensor | None = None
    entropy: torch.Tensor | None = None
    slots: torch.Tensor | None = None
    emit: torch.Tensor | None = None
    salt: torch.Tensor | None = None
    stops: torch.Tensor | None = None
    start: torch.Tensor | None = None


def select(rows: CanvasRows, index: torch.Tensor | None) -> CanvasRows:
    if index is None:
        return rows
    values = {f.name: getattr(rows, f.name) for f in fields(rows)}
    return replace(rows, **{k: v[index] for k, v in values.items() if v is not None})
