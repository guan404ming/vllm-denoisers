from dataclasses import replace

import torch

from denoisers.canvas.rows import CanvasRows, StepResult
from denoisers.ops.sampling import given, sample, select_unmask
from denoisers.ops.stats import row_stats
from denoisers.policies.decode.interface import DecodePolicy, registry


def apply_edits(
    block: torch.Tensor,
    old: torch.Tensor,
    original: torch.Tensor,
    mask_id: int,
    delete_id: int,
    split_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    n, width = block.shape
    is_delete = block == delete_id
    is_split = block == split_id
    keep = ~is_delete & ~is_split
    count = torch.where(is_delete, 0, torch.where(is_split, 2, 1))
    start = count.cumsum(dim=1) - count
    spill = 2 * width
    out = block.new_full((n, spill + 1), mask_id)
    out_original = torch.zeros(n, spill + 1, dtype=torch.bool, device=block.device)
    keep_at = torch.where(keep, start, spill)
    out.scatter_(1, keep_at, torch.where(keep, block, mask_id))
    out_original.scatter_(1, keep_at, keep & original)
    split_at = torch.where(is_split, start + 1, spill)
    out.scatter_(1, split_at, torch.where(is_split, old, mask_id))
    out_original.scatter_(1, split_at, is_split & original)
    return out[:, :width], out_original[:, :width]


@registry.register("indel")
class Indel(DecodePolicy):
    commit_logits = False

    def __init__(
        self,
        mask_id: int,
        delete_id: int,
        split_id: int,
        steps: int,
        threshold: float,
        editing_threshold: float,
        max_post_steps: int,
        max_steps_per_block: int,
        max_resamples: int = 5,
        temperature: float = 0.0,
        history_length: int | None = None,
    ):
        self.mask_id = mask_id
        self.temperature = temperature
        self.delete_id = delete_id
        self.split_id = split_id
        self.steps = steps
        self.threshold = threshold
        self.editing_threshold = editing_threshold
        self.max_post_steps = max_post_steps
        self.max_steps_per_block = max_steps_per_block
        self.history_length = max_steps_per_block if history_length is None else history_length
        if self.history_length < 1:
            raise ValueError("Indel needs a positive history_length.")
        self.max_resamples = max_resamples

    @classmethod
    def from_config(cls, config, model) -> "Indel":
        return cls(
            mask_id=config.mask_token_id,
            temperature=config.temperature,
            delete_id=config.delete_token_id,
            split_id=config.split_token_id,
            steps=config.denoise_steps,
            threshold=given(config.confidence_threshold, 0.5),
            editing_threshold=given(config.editing_threshold, 0.0),
            max_post_steps=config.max_post_steps,
            max_steps_per_block=config.max_steps_per_block,
            history_length=getattr(config, "history_length", None),
        )

    def _escape_loops(self, written, before, logits, p0, rows, allowed):
        n = written.shape[0]
        history, history_len = rows.history, rows.history_len
        device = written.device
        valid = torch.arange(history.shape[1], device=device) < history_len.unsqueeze(1)

        def repeats(state):
            return ((history == state.unsqueeze(1)).all(dim=2) & valid).any(dim=1)

        changed = written != before
        order = torch.where(changed, p0, torch.inf).argsort(dim=1)
        count = changed.sum(dim=1)
        index = torch.arange(n, device=device)
        active = repeats(written) & allowed
        for r in range(min(self.max_resamples, order.shape[1])):
            take = active & (r < count)
            pos = order[:, r]
            row_logits = logits[index, pos].float()
            row_logits[index, written[index, pos]] = -torch.inf
            row_logits[:, self.mask_id] = -torch.inf
            replaced = written.clone()
            replaced[index, pos] = row_logits.argmax(dim=1)
            written = torch.where(take.unsqueeze(1), replaced, written)
            active = take & repeats(written)
        history = history.clone()
        history[index, history_len % history.shape[1]] = written
        return written, history, history_len + 1

    def __call__(self, logits: torch.Tensor, rows: CanvasRows) -> tuple[CanvasRows, StepResult]:
        canvas, original = rows.canvas, rows.original
        block = canvas.shape[1]
        positions = torch.arange(block, device=canvas.device)
        prompt = positions < rows.shift.unsqueeze(1)
        masked = (canvas == self.mask_id) & ~prompt
        new_masks = (masked & ~original).sum(dim=1)
        final_round = rows.post_steps >= self.max_post_steps
        no_edit_ops = final_round | (rows.width < block)

        if self.temperature > 0:
            x0, p0 = sample(logits, self.mask_id, self.temperature)
            alt = row_stats(logits, self.mask_id, (self.delete_id, self.split_id)).alt
        else:
            stats = row_stats(logits, self.mask_id, (self.delete_id, self.split_id))
            x0, p0, alt = stats.x0, stats.p0, stats.alt
        initial = block - rows.shift
        quota = initial // self.steps + (rows.step < initial % self.steps).long()
        chosen = select_unmask(masked, p0, quota + new_masks, self.threshold)
        scheduled = (rows.step < self.steps) & ~final_round
        mt2 = torch.where(scheduled.unsqueeze(1), chosen, masked)
        editable = ~masked & ~prompt
        t2t = editable & (p0 > self.editing_threshold) & (x0 != canvas)

        is_edit_op = (x0 == self.delete_id) | (x0 == self.split_id)
        token = torch.where(no_edit_ops.unsqueeze(1) & is_edit_op, alt, x0)
        written = torch.where(mt2 | t2t, token, canvas)
        written, history, history_len = self._escape_loops(
            written, canvas, logits, p0, rows, ~final_round
        )
        edited, original_after = apply_edits(
            written, canvas, original, self.mask_id, self.delete_id, self.split_id
        )

        masked_after = (edited == self.mask_id) & ~prompt
        post_next = torch.where((masked_after & original_after).any(dim=1), 0, rows.post_steps + 1)
        settled = ~masked_after.any(dim=1) & (
            (edited == canvas).all(dim=1) | (post_next > self.max_post_steps)
        )
        done = settled | (rows.step + 1 >= self.max_steps_per_block)
        after = replace(
            rows,
            canvas=edited,
            original=original_after,
            post_steps=post_next,
            history=history,
            history_len=history_len,
        )
        return after, StepResult(done)
