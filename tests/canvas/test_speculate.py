import numpy as np
import pytest
import torch

from denoisers.canvas.speculate import AHEAD, FOLLOW, plan, speculate
from denoisers.canvas.state import CanvasState
from denoisers.policies.decode.dream import Dream
from denoisers.policies.decode.fast_dllm import FastDLLM
from denoisers.policies.decode.llada import LLaDA
from denoisers.policies.decode.low_confidence import LowConfidence

BLOCK = 4
MASK = 9
VOCAB = 6
BLOCKS = 12
THRESHOLD = 0.6


def model(prefix: list[int], cells: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    start = len(prefix)
    logits = torch.empty(BLOCK, VOCAB)
    for i in range(BLOCK):
        seen = sum((j + 1) * t for j, t in enumerate(prefix[-BLOCK:]))
        base = torch.Generator().manual_seed(start + i + 31 * seen)
        noise = torch.Generator().manual_seed(hash((start, i, *cells.tolist())) % (1 << 31))
        logits[i] = 3 * torch.randn(VOCAB, generator=base) + torch.randn(VOCAB, generator=noise)
    return torch.softmax(logits, -1).max(-1)


POLICIES = {
    "low_confidence": LowConfidence(MASK, BLOCK, BLOCK, THRESHOLD),
    "llada": LLaDA(MASK, BLOCK, 2, BLOCK),
    "dream": Dream(MASK, BLOCK, BLOCK),
    "fast_dllm": FastDLLM(MASK, BLOCK, 2, THRESHOLD),
}


def reference(policy, seeded: bool = False) -> tuple[list[int], int]:
    zero, full = torch.zeros(1, dtype=torch.int64), torch.full((1,), BLOCK)
    out, forwards, start = [], 0, torch.full((1, BLOCK), MASK)
    for _ in range(BLOCKS):
        canvas, done, step = start, False, 0
        while not done:
            confidence, tokens = model(out, canvas[0])
            canvas, done = policy.move(
                canvas, torch.tensor([step]), zero, full, tokens[None], confidence[None]
            )
            step += 1
            forwards += 1
        start = torch.full((1, BLOCK), MASK)
        if seeded:
            start[0, 0] = model(out, canvas[0])[1][-1]
        out += canvas[0].tolist()
        forwards += 1
    return out, forwards


def decode(
    widths, depth: int = 1, policy=None, stop: int | None = None, leave=True, seeded=False
) -> tuple[list[int], int, int]:
    policy = policy or POLICIES["low_confidence"]
    state = CanvasState(1, BLOCK, MASK, torch.device("cpu"), levels=depth)
    state.begin(0, torch.empty(0, dtype=torch.int64), 0)
    state.stop_at(0, {stop} if leave else ())
    slots = torch.arange(1)
    drafts = torch.full(((2 * depth + 2) * BLOCK,), MASK)
    out, forwards, advanced = [], 0, 0
    while len(out) < BLOCKS * BLOCK and stop not in out:
        flags = plan(
            state.mode.numpy().copy(),
            state.committing.numpy().copy(),
            state.ahead_done.numpy().copy(),
            np.array([widths[forwards % len(widths)] * BLOCK]),
            BLOCK,
        )
        confidences = torch.zeros(len(flags.groups), 1, BLOCK)
        choices = torch.zeros(len(flags.groups), 1, BLOCK, dtype=torch.int64)
        for group, (pick, local) in enumerate(flags.groups):
            if not pick[0]:
                continue
            at = int(local[0])
            seen = max(group - FOLLOW + 1, int(group == AHEAD), 0)
            prefix = out + drafts[: seen * BLOCK].tolist()
            confidences[group, 0], choices[group, 0] = model(
                prefix, drafts[at * BLOCK : (at + 1) * BLOCK]
            )
        seed = model(out, drafts[:BLOCK])[1][-1:] if seeded else None
        sampled, num_sampled, rows, advance = speculate(
            state, slots, flags, torch.from_numpy, confidences, choices, policy, seed=seed
        )
        drafts = rows[0]
        out += sampled[0, : int(num_sampled[0])].tolist()
        forwards += 1
        advanced += int(advance[0])
    return out, forwards, advanced - int(state.step[0])


@pytest.mark.parametrize("name", list(POLICIES))
@pytest.mark.parametrize(
    "widths",
    [(1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (4, 1, 3, 8, 2, 5, 2, 7, 3, 6, 4, 1, 1)],
)
def test_speculation_matches_reference_decoder(widths, name):
    expected, plain = reference(POLICIES[name])
    depth = max(1, (max(widths) - 1) // 2)
    out, _, advanced = decode(widths, depth, POLICIES[name])
    assert out[: len(expected)] == expected
    assert advanced == plain


@pytest.mark.parametrize("name", ["dream", "fast_dllm"])
@pytest.mark.parametrize("widths", [(1,), (2,), (2, 1, 2, 2, 1)])
def test_seed_guess_matches_seeded_reference(widths, name):
    expected, plain = reference(POLICIES[name], seeded=True)
    out, _, advanced = decode(widths, 1, POLICIES[name], seeded=True)
    assert out[: len(expected)] == expected
    assert advanced == plain


def test_wider_layouts_need_fewer_forwards():
    _, plain = reference(POLICIES["low_confidence"])
    counts = [decode((width,), 3)[1] for width in range(1, 9)]
    assert counts[0] == plain
    assert counts[3] < counts[1] < counts[0]
    assert counts[7] <= counts[5] <= counts[3] <= counts[2] < counts[0]


@pytest.mark.parametrize("name", list(POLICIES))
def test_final_prefix_exit_keeps_the_text_before_the_stop(name):
    expected, _ = reference(POLICIES[name])
    saved = 0
    for width in (1, 2, 4):
        for stop in set(expected):
            cut = expected.index(stop) + 1
            out, forwards, _ = decode((width,), 1, POLICIES[name], stop)
            _, slow, _ = decode((width,), 1, POLICIES[name], stop, leave=False)
            assert out[:cut] == expected[:cut]
            assert forwards <= slow
            saved += slow - forwards
    assert saved
