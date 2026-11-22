import torch

from denoisers.canvas.rows import CanvasRows

STOPS = 8


class CanvasState:
    def __init__(
        self,
        max_num_reqs: int,
        block_length: int,
        mask_id: int,
        device: torch.device,
        history_length: int = 64,
        hidden_size: int | None = None,
    ):
        self.block_length = block_length
        self.mask_id = mask_id
        shape = (max_num_reqs, block_length)
        self.canvas = torch.full(shape, mask_id, dtype=torch.int64, device=device)
        self.original = torch.ones(shape, dtype=torch.bool, device=device)
        self.step = torch.zeros(max_num_reqs, dtype=torch.int64, device=device)
        self.shift = torch.zeros(max_num_reqs, dtype=torch.int64, device=device)
        self.start = torch.zeros(max_num_reqs, dtype=torch.int64, device=device)
        self.width = torch.full((max_num_reqs,), block_length, dtype=torch.int64, device=device)
        self.post_steps = torch.zeros(max_num_reqs, dtype=torch.int64, device=device)
        self.salt = torch.zeros(max_num_reqs, dtype=torch.int64, device=device)
        self.begun = 0
        self.stops = torch.full((max_num_reqs, STOPS), -1, dtype=torch.int64, device=device)
        self.history = torch.zeros(
            (max_num_reqs, history_length, block_length), dtype=torch.int64, device=device
        )
        self.history_len = torch.zeros(max_num_reqs, dtype=torch.int64, device=device)
        self.prefilling = torch.zeros(max_num_reqs, dtype=torch.bool, device=device)
        self.committing = torch.zeros(max_num_reqs, dtype=torch.bool, device=device)
        self.active = torch.arange(block_length, device=device).repeat(max_num_reqs, 1)
        self.active_len = torch.full(
            (max_num_reqs,), block_length, dtype=torch.int32, device=device
        )
        self.soft = (
            None
            if hidden_size is None
            else torch.zeros((*shape, hidden_size), dtype=torch.float32, device=device)
        )

    def reset(self, slot: int) -> None:
        self.canvas[slot] = self.mask_id
        self.original[slot] = True
        self.step[slot] = 0
        self.shift[slot] = 0
        self.start[slot] = 0
        self.width[slot] = self.block_length
        self.post_steps[slot] = 0
        self.history_len[slot] = 0
        self.prefilling[slot] = True
        self.committing[slot] = False
        self.active[slot] = torch.arange(self.block_length, device=self.active.device)
        self.active_len[slot] = self.block_length
        if self.soft is not None:
            self.soft[slot] = 0

    def stop_at(self, slot: int, tokens) -> None:
        ids = sorted(token for token in tokens if token is not None)[:STOPS]
        self.stops[slot] = -1
        self.stops[slot, : len(ids)] = torch.tensor(ids, dtype=torch.int64)

    def begin(self, slot: int, prompt_tail: torch.Tensor, start: int) -> None:
        self.reset(slot)
        shift = prompt_tail.numel()
        self.canvas[slot, :shift] = prompt_tail
        self.original[slot, :shift] = False
        self.shift[slot] = shift
        self.start[slot] = start
        self.prefilling[slot] = False
        self.begun += 1
        self.salt[slot] = self.begun << 20

    def truncate(self, slots: torch.Tensor, widths: torch.Tensor) -> None:
        self.width[slots] = widths.to(self.width.dtype)
        beyond = torch.arange(self.block_length, device=widths.device) >= widths.unsqueeze(1)
        self.canvas[slots] = torch.where(beyond, 0, self.canvas[slots])
        self.original[slots] &= ~beyond

    def gather(self, slots: torch.Tensor) -> CanvasRows:
        return CanvasRows(
            canvas=self.canvas[slots],
            original=self.original[slots],
            step=self.step[slots],
            shift=self.shift[slots],
            width=self.width[slots],
            post_steps=self.post_steps[slots],
            history=self.history[slots],
            history_len=self.history_len[slots],
            committing=self.committing[slots],
            soft=None if self.soft is None else self.soft[slots],
            slots=slots,
            salt=self.salt[slots],
            stops=self.stops[slots],
            start=self.start[slots],
        )

    def scatter(self, slots: torch.Tensor, rows: CanvasRows) -> None:
        self.canvas[slots] = rows.canvas
        self.original[slots] = rows.original
        self.step[slots] = rows.step
        self.shift[slots] = rows.shift
        self.width[slots] = rows.width
        self.post_steps[slots] = rows.post_steps
        self.history[slots] = rows.history
        self.history_len[slots] = rows.history_len
        self.committing[slots] = rows.committing
        self.salt[slots] = rows.salt
        self.start[slots] = rows.start
        if self.soft is not None:
            self.soft[slots] = rows.soft
