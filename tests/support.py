import torch

from denoisers.canvas.rows import CanvasRows

HISTORY = 8


def make_rows(
    canvas: torch.Tensor, step=None, shift=None, post_steps=None, mask_id=None
) -> CanvasRows:
    n, block = canvas.shape
    zeros = torch.zeros(n, dtype=torch.int64)
    original = torch.ones_like(canvas, dtype=torch.bool) if mask_id is None else canvas == mask_id
    return CanvasRows(
        canvas=canvas,
        original=original,
        step=zeros if step is None else torch.as_tensor(step),
        shift=zeros if shift is None else torch.as_tensor(shift),
        width=torch.full((n,), block, dtype=torch.int64),
        post_steps=zeros if post_steps is None else torch.as_tensor(post_steps),
        history=torch.zeros(n, HISTORY, block, dtype=torch.int64),
        history_len=zeros.clone(),
        committing=torch.zeros(n, dtype=torch.bool),
    )
