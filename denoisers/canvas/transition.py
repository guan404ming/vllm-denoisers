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
    history: torch.Tensor
    history_len: torch.Tensor
    committing: torch.Tensor
    soft: torch.Tensor | None = None
    entropy: torch.Tensor | None = None
    slots: torch.Tensor | None = None
    emit: torch.Tensor | None = None
    salt: torch.Tensor | None = None
    stops: torch.Tensor | None = None


def final_prefix(
    canvas: torch.Tensor, shift: torch.Tensor, stops: torch.Tensor, mask_id: int
) -> torch.Tensor:
    cells = torch.arange(canvas.shape[1], device=canvas.device)
    stop = (canvas.unsqueeze(2) == stops.unsqueeze(1)).any(2)
    wall = (stop | (canvas == mask_id)) & (cells >= shift.unsqueeze(1))
    first = wall.int().argmax(1, keepdim=True)
    return (wall & stop).gather(1, first).squeeze(1)


def fresh_rows(
    rows: CanvasRows, mask_id: int, policy, logits: torch.Tensor
) -> CanvasRows:
    canvas = policy.init_canvas(torch.full_like(rows.canvas, mask_id), rows.slots)
    seed = None if logits is None else policy.next_seed(logits)
    if seed is not None:
        canvas[:, 0] = seed
    return CanvasRows(
        canvas=canvas,
        original=torch.ones_like(rows.original),
        step=torch.zeros_like(rows.step),
        shift=torch.zeros_like(rows.shift),
        width=torch.full_like(rows.width, rows.canvas.shape[1]),
        post_steps=torch.zeros_like(rows.post_steps),
        history=torch.zeros_like(rows.history),
        history_len=torch.zeros_like(rows.history_len),
        committing=torch.zeros_like(rows.committing),
        soft=None if rows.soft is None else torch.zeros_like(rows.soft),
        slots=rows.slots,
        salt=None if rows.salt is None else rows.salt + 1,
        stops=rows.stops,
    )


def _select(committing: torch.Tensor, reset: torch.Tensor, value: torch.Tensor):
    flag = committing.view(-1, *([1] * (value.dim() - 1)))
    return torch.where(flag, reset, value)


def advance(
    rows: CanvasRows, logits: torch.Tensor | None, policy, mask_id: int
) -> tuple[CanvasRows, torch.Tensor, torch.Tensor]:
    block = rows.canvas.shape[1]
    if logits is None:
        after, done = rows, torch.zeros_like(rows.committing)
    else:
        after, done = policy(logits, rows)
        if rows.stops is not None and policy.stop_exit:
            done = done | final_prefix(after.canvas, rows.shift, rows.stops, mask_id)
    now = torch.zeros_like(rows.committing) if after.emit is None else after.emit
    committing = rows.committing | now

    offsets = torch.arange(block, device=rows.canvas.device)
    keep = (rows.width - rows.shift).clamp(min=0)
    source = torch.where(now.unsqueeze(1), after.canvas, rows.canvas)
    packed = source.gather(1, (offsets + rows.shift.unsqueeze(1)).clamp(max=block - 1))
    emit = committing.unsqueeze(1) & (offsets < keep.unsqueeze(1))
    sampled = torch.where(emit, packed, 0)
    num_sampled = torch.where(committing, keep, 0)

    fresh = fresh_rows(rows, mask_id, policy, logits)
    result = CanvasRows(
        canvas=_select(committing, fresh.canvas, after.canvas),
        original=_select(committing, fresh.original, after.original),
        step=_select(committing, fresh.step, rows.step + 1),
        shift=_select(committing, fresh.shift, rows.shift),
        width=_select(committing, fresh.width, rows.width),
        post_steps=_select(committing, fresh.post_steps, after.post_steps),
        history=_select(committing, fresh.history, after.history),
        history_len=_select(committing, fresh.history_len, after.history_len),
        committing=~committing & done,
        soft=None if after.soft is None else _select(committing, fresh.soft, after.soft),
        entropy=after.entropy,
        slots=rows.slots,
        salt=None if rows.salt is None else _select(committing, fresh.salt, rows.salt),
        stops=rows.stops,
    )
    return result, sampled, num_sampled
