import numpy as np
import pytest
import torch
from support import make_rows

from denoisers.canvas.transition import advance, final_prefix
from denoisers.policies.decode.llada import LLaDA
from denoisers.policies.decode.low_confidence import LowConfidence
from denoisers.runtime.batch import free_ranges, prefill_block_ranges

MASK = 99
BLOCK = 32


def triton_mask(seq_len, ranges):
    q = torch.arange(seq_len).unsqueeze(1)
    k = torch.arange(seq_len).unsqueeze(0)
    mask = k <= q
    for start, end in ranges:
        mask |= (q >= start) & (q <= end) & (k >= start) & (k <= end)
    return mask


def hf_block_causal(seq_len, block):
    blocks = torch.arange(seq_len) // block
    return blocks.unsqueeze(0) <= blocks.unsqueeze(1)


@pytest.mark.parametrize("seq_len", [1, 2, 31, 32, 33, 49, 64, 100])
def test_prefill_ranges_match_hf_block_causal(seq_len):
    ranges = prefill_block_ranges(seq_len, BLOCK)
    assert torch.equal(triton_mask(seq_len, ranges), hf_block_causal(seq_len, BLOCK))


def test_free_ranges_open_only_rows_that_finished_prefill():
    blocks = prefill_block_ranges(40, BLOCK)
    ranges = free_ranges({0: blocks}, np.array([True, False, True]))
    assert ranges[0] == blocks and 2 not in ranges
    assert triton_mask(40, ranges[1]).all()


@pytest.mark.parametrize("shift", [0, 1, 17, 31])
def test_block_lifecycle_emits_and_aligns(shift):
    gen = torch.Generator().manual_seed(shift)
    prompt_len = 64 + shift
    tail = torch.randint(0, MASK, (shift,), generator=gen)
    canvas = torch.full((1, BLOCK), MASK)
    canvas[0, :shift] = tail
    rows = make_rows(canvas, shift=[shift], mask_id=MASK)
    policy = LowConfidence(MASK, BLOCK, BLOCK, 2.0)
    num_computed = prompt_len
    emitted = None
    for _ in range(BLOCK + 1):
        logits = torch.randn(1, BLOCK, MASK + 1, generator=gen)
        logits[..., MASK] = -torch.inf
        final = rows.canvas.clone()
        rows, sampled, num_sampled, _ = advance(rows, logits, policy, MASK)
        num_computed += int(num_sampled)
        if num_sampled.item():
            emitted = sampled[0, : int(num_sampled)]
            break
    assert emitted is not None
    assert torch.equal(emitted, final[0, shift:])
    assert torch.equal(final[0, :shift], tail)
    assert num_computed == prompt_len - shift + BLOCK
    assert num_computed % BLOCK == 0
    assert int(rows.shift) == 0 and int(rows.step) == 0
    assert (rows.canvas == MASK).all()
    assert rows.original.all() and int(rows.post_steps) == 0 and int(rows.history_len) == 0


def test_commit_without_logits_emits_the_canvas():
    canvas = torch.arange(BLOCK).unsqueeze(0)
    rows = make_rows(canvas, mask_id=MASK)
    rows.committing = torch.tensor([True])
    after, sampled, num_sampled, _ = advance(
        rows, None, LowConfidence(MASK, BLOCK, BLOCK, 2.0), MASK
    )
    assert int(num_sampled) == BLOCK
    assert torch.equal(sampled[0], canvas[0])
    assert (after.canvas == MASK).all() and not after.committing.item()


def test_final_prefix_needs_every_cell_before_the_first_stop():
    stop, other = 7, 8
    canvas = torch.tensor(
        [
            [1, 2, stop, MASK, MASK],
            [1, MASK, stop, MASK, MASK],
            [1, 2, 3, MASK, MASK],
            [stop, MASK, 3, other, MASK],
            [MASK, 2, other, MASK, stop],
        ]
    )
    shift = torch.tensor([0, 0, 0, 1, 1])
    stops = torch.tensor([[stop, -1], [stop, -1], [stop, -1], [stop, other], [stop, other]])
    found = final_prefix(canvas, shift, stops, MASK)
    assert found.tolist() == [True, False, False, False, True]


@pytest.mark.parametrize(
    ("remaining", "second_token", "counts"),
    [([2, 1], MASK, [2, 1]), ([2, 2], MASK, [0, 0]), ([2, 4], 8, [0, 0])],
)
def test_terminal_prefix_waits_for_the_whole_batch(remaining, second_token, counts):
    canvas = torch.tensor([[4, MASK, 8, MASK], [3, MASK, second_token, MASK]])
    rows = make_rows(canvas, shift=[1, 1], mask_id=MASK)
    logits = torch.zeros(2, 4, MASK + 1)
    logits[:, 1, 7] = 20
    policy = LLaDA(MASK, 4, 4, 4)
    _, sampled, emitted, _ = advance(rows, logits, policy, MASK, torch.tensor(remaining))
    assert emitted.tolist() == counts
    expected = [[7, 8, 0, 0], [7, 0, 0, 0]] if counts[0] else [[0] * 4, [0] * 4]
    assert sampled.tolist() == expected
