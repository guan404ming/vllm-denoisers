import functools
from typing import NamedTuple

import numpy as np
import torch

from denoisers.canvas.state import CanvasState
from denoisers.canvas.transition import final_prefix

PLAIN, STEP, TREE = 0, 1, 2
MAIN, GUESS, AHEAD, FOLLOW = range(4)


class Plan(NamedTuple):
    live: np.ndarray
    step: np.ndarray
    ahead: np.ndarray
    grow: np.ndarray
    wide: np.ndarray
    target: np.ndarray
    groups: tuple[tuple[np.ndarray, np.ndarray], ...]


def plan(
    mode: np.ndarray,
    commit: np.ndarray,
    next_done: np.ndarray,
    widths: np.ndarray,
    block: int,
) -> Plan:
    depth = next_done.shape[1]
    count = widths // block
    fills = np.maximum(mode - 1, 0)
    live = count > fills
    step = (mode > PLAIN) & (count >= 2 * fills + 2) & ~commit
    ahead = (mode == PLAIN) & commit & (count >= 2)
    wide = live & (fills > 0)
    levels = np.arange(1, depth + 1)[:, None]
    grow = wide & (levels <= fills) & (count > fills + levels) & ~next_done.T
    target = np.where(
        count >= 3,
        1 + np.minimum((count - 1) // 2, depth),
        np.where(count == 2, STEP, PLAIN),
    )
    groups = (
        (live & ~commit, fills),
        (step, 2 * fills + 1),
        (ahead, np.ones_like(fills)),
        *((grow[level], fills + level + 1) for level in range(depth)),
    )
    return Plan(live, step, ahead, grow, wide, target, groups)


@functools.cache
def layout(depth: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    width = 2 * depth + 2
    source = torch.full((depth + 2, width), 2)
    offset = torch.zeros((depth + 2, width), dtype=torch.int64)
    source[PLAIN, 0] = 0
    for fills in range(depth + 1):
        row = fills + 1
        for cell in range(width):
            if cell < fills:
                source[row, cell] = 3 + cell
            elif cell == fills:
                source[row, cell] = 0
            elif cell <= 2 * fills:
                source[row, cell] = 3 + depth + cell - fills - 1
            elif cell == 2 * fills + 1:
                source[row, cell] = 1
            offset[row, cell] = 0 if cell < fills else fills if cell <= 2 * fills else cell
    return source.to(device), offset.to(device)


def pack(flags: Plan) -> np.ndarray:
    return np.stack(
        [flags.live, flags.wide, flags.step, flags.ahead, *flags.grow, flags.target]
    ).astype(np.int64)


def speculate(
    state: CanvasState,
    slots: torch.Tensor,
    flags: Plan,
    gpu,
    confidences: torch.Tensor,
    choices: torch.Tensor,
    policy,
    later: tuple[torch.Tensor, torch.Tensor] | None = None,
    seed: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    return settle(state, slots, gpu(pack(flags)), confidences, choices, policy, later, seed)


def settle(
    state: CanvasState,
    slots: torch.Tensor,
    packed: torch.Tensor,
    confidences: torch.Tensor,
    choices: torch.Tensor,
    policy,
    later: tuple[torch.Tensor, torch.Tensor] | None = None,
    seed: torch.Tensor | None = None,
) -> tuple[torch.Tensor, ...]:
    block, mask_id = state.block_length, state.mask_id
    canvas, steps = state.canvas[slots], state.step[slots]
    shift, width, closing = state.shift[slots], state.width[slots], state.committing[slots]
    cells = torch.arange(block, device=canvas.device)
    blank = torch.full_like(canvas, mask_id)
    start = (
        blank
        if seed is None
        else blank.scatter(1, cells[:1].expand(len(seed), 1), seed.unsqueeze(1))
    )
    next_scores, next_choices = later or (confidences, choices)
    zero, full = torch.zeros_like(shift), torch.full_like(width, block)
    stops = state.stops[slots]

    def move(canvas, steps, shift, width, tokens, score):
        after, done = policy.move(canvas, steps, shift, width, tokens, score)
        return after, done | final_prefix(after, shift, stops, mask_id)

    def emit(flag):
        packed = canvas.gather(1, (cells + shift.unsqueeze(1)).clamp(max=block - 1))
        kept = (width - shift).clamp(min=0)
        return (
            torch.where(flag.unsqueeze(1) & (cells < kept.unsqueeze(1)), packed, 0),
            torch.where(flag, kept, 0),
        )

    ahead, ahead_steps, ahead_done = (
        state.ahead[slots],
        state.ahead_step[slots],
        state.ahead_done[slots],
    )
    fed = state.fill[slots]
    depth = ahead.shape[1]
    flags = packed.bool()
    alive, wide, stepping, opening = flags[0], flags[1], flags[2], flags[3]
    fresh, moved = alive & closing & ~wide, alive & ~closing
    sampled, num_sampled = emit(fresh)
    after, done = move(canvas, steps, shift, width, choices[MAIN], confidences[MAIN])
    canvas = torch.where(fresh.unsqueeze(1), start, torch.where(moved.unsqueeze(1), after, canvas))
    steps = torch.where(fresh, 0, torch.where(moved, steps + 1, steps))
    shift = torch.where(fresh, 0, shift)
    width = torch.where(fresh, block, width)
    closing = torch.where(moved, done, closing & ~fresh)
    next_score, next_tokens = next_scores[MAIN], next_choices[MAIN]
    advance = (fresh | moved).int()

    for group, hit in ((GUESS, stepping), (AHEAD, opening)):
        if group == GUESS:
            hit = hit & (canvas == state.guess[slots]).all(1) & ~closing
        elif seed is not None:
            hit = hit & (canvas == state.guess[slots]).all(1)
        after, done = move(canvas, steps, shift, width, choices[group], confidences[group])
        canvas = torch.where(hit.unsqueeze(1), after, canvas)
        steps = torch.where(hit, steps + 1, steps)
        advance = advance + hit
        closing = torch.where(hit, done, closing)
        next_score = torch.where(hit.unsqueeze(1), next_scores[group], next_score)
        next_tokens = torch.where(hit.unsqueeze(1), next_choices[group], next_tokens)

    fills = [torch.where((canvas == mask_id) & ~closing.unsqueeze(1), next_tokens, canvas)]
    for level in range(depth):
        growing = flags[4 + level]
        after, done = move(
            ahead[:, level],
            ahead_steps[:, level],
            zero,
            full,
            choices[FOLLOW + level],
            confidences[FOLLOW + level],
        )
        ahead[:, level] = torch.where(growing.unsqueeze(1), after, ahead[:, level])
        ahead_steps[:, level] += growing
        ahead_done[:, level] = torch.where(growing, done, ahead_done[:, level])
        fills.append(fed[:, level + 1] if level + 1 < depth else ahead[:, level])
        guess = torch.where(after == mask_id, choices[FOLLOW + level], after)
        fills[-1] = torch.where(growing.unsqueeze(1), guess, fills[-1])
    fills = torch.stack(fills, 1)
    same = (fills[:, :depth] == fed).all(2)
    valid = wide.unsqueeze(1) & same.cumprod(1).bool()
    early = wide & closing & (canvas == fed[:, 0]).all(1)
    early_sampled, early_num = emit(early)
    sampled = torch.where(early.unsqueeze(1), early_sampled, sampled)
    num_sampled = torch.where(early, early_num, num_sampled)
    advance = advance + torch.where(early, ahead_steps[:, 0] + 1, 0)
    canvas = torch.where(early.unsqueeze(1), ahead[:, 0], canvas)
    steps = torch.where(early, ahead_steps[:, 0], steps)
    closing = torch.where(early, ahead_done[:, 0], closing)
    shift = torch.where(early, 0, shift)
    width = torch.where(early, block, width)
    next_score = torch.where(early.unsqueeze(1), next_scores[FOLLOW], next_score)
    next_tokens = torch.where(early.unsqueeze(1), next_choices[FOLLOW], next_tokens)

    def pull(values, empty):
        moved = torch.cat([values[:, 1:], torch.full_like(values[:, :1], empty)], 1)
        return torch.where(early.view(-1, *[1] * (values.dim() - 1)), moved, values)

    ahead, ahead_steps = pull(ahead, mask_id), pull(ahead_steps, 0)
    ahead_done, fills, valid = pull(ahead_done, False), pull(fills, mask_id), pull(valid, False)
    fills = fills[:, :depth]

    masked = (canvas == mask_id) & ~closing.unsqueeze(1)
    guess = policy.move(canvas, steps, shift, width, next_tokens, next_score)[0]
    guess = torch.where(closing.unsqueeze(1), canvas if seed is None else start, guess)
    target = packed[4 + depth]
    mode = torch.where(masked.any(1) | (target >= TREE), target, PLAIN).to(state.mode.dtype)
    levels = torch.arange(1, depth + 1, device=canvas.device)
    keep = valid & (levels < mode.unsqueeze(1))
    ahead = torch.where(keep.unsqueeze(2), ahead, blank.unsqueeze(1))

    state.canvas[slots], state.step[slots], state.committing[slots] = canvas, steps, closing
    state.shift[slots], state.width[slots] = shift, width
    state.salt[slots] += fresh | early
    state.fill[slots], state.guess[slots], state.mode[slots] = fills, guess, mode
    state.ahead[slots] = ahead
    state.ahead_step[slots] = torch.where(keep, ahead_steps, 0)
    state.ahead_done[slots] = keep & ahead_done
    source, _ = layout(depth, canvas.device)
    parts = torch.cat([torch.stack([canvas, guess, start], 1), fills, ahead], 1)
    index = source[mode.long()]
    drafts = parts.gather(1, index.unsqueeze(2).expand(-1, -1, block)).flatten(1)
    return sampled, num_sampled, drafts, advance
