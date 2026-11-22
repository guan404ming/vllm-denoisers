import torch
from support import make_rows

from denoisers.canvas.state import CanvasState
from denoisers.canvas.transition import advance
from denoisers.ops.sampling import sample
from denoisers.policies.decode.indel import Indel
from denoisers.policies.decode.low_confidence import LowConfidence
from denoisers.runtime.batch import aligned_prefill_chunk

MASK = 9
BLOCK = 8


def test_sample_with_temperature_draws_but_never_the_mask():
    logits = torch.full((3, 2, MASK + 1), -1e9)
    logits[:, :, 3] = 0.0
    logits[:, :, 5] = 0.0
    logits[:, :, MASK] = 50.0
    x0, p0 = sample(logits, MASK, 0.7)
    assert (x0 != MASK).all()
    assert set(x0.flatten().tolist()) <= {3, 5}
    scaled = torch.softmax(logits / 0.7, -1)
    assert torch.allclose(p0, scaled.gather(-1, x0.unsqueeze(-1)).squeeze(-1))
    x0_greedy, _ = sample(logits, MASK)
    assert (x0_greedy == 3).all()


def test_truncated_canvas_emits_only_valid_positions():
    state = CanvasState(2, BLOCK, MASK, torch.device("cpu"))
    state.begin(0, torch.tensor([1, 2]), 0)
    state.begin(1, torch.tensor([4]), 0)
    state.truncate(torch.tensor([0]), torch.tensor([5], dtype=torch.int32))
    assert state.width.tolist() == [5, BLOCK]
    assert (state.canvas[0, 5:] == 0).all() and not state.original[0, 5:].any()
    rows = state.gather(torch.tensor([0, 1]))
    policy = LowConfidence(MASK, BLOCK, 1, 2.0)
    logits = torch.full((2, BLOCK, MASK + 1), -1e9)
    logits[..., 7] = 0.0
    logits[0, 5:, 0] = 1e9
    rows, sampled, num_sampled = advance(rows, logits, policy, MASK)
    assert num_sampled.tolist() == [0, 0]
    assert rows.canvas[0].tolist() == [1, 2, 7, 7, 7, 0, 0, 0]
    rows, sampled, num_sampled = advance(rows, logits, policy, MASK)
    assert num_sampled.tolist() == [3, 7]
    assert sampled[0, :3].tolist() == [7, 7, 7]
    assert rows.width.tolist() == [BLOCK, BLOCK]


def test_aligned_prefill_chunk():
    assert aligned_prefill_chunk(0, 48, 100, 32) == 32
    assert aligned_prefill_chunk(32, 48, 100, 32) == 32
    assert aligned_prefill_chunk(64, 48, 100, 32) == 48
    assert aligned_prefill_chunk(0, 20, 100, 32) == 0
    assert aligned_prefill_chunk(96, 4, 100, 32) == 4
    assert aligned_prefill_chunk(100, 32, 100, 32) == 32


def test_make_rows_keeps_width_full():
    rows = make_rows(torch.full((1, BLOCK), MASK))
    assert rows.width.tolist() == [BLOCK]


def test_truncated_block_blocks_indel_edit_ops():
    delete, split = MASK + 1, MASK + 2
    canvas = torch.full((2, BLOCK), MASK)
    canvas[:, 0] = 1
    rows = make_rows(canvas, shift=[1, 1], mask_id=MASK)
    rows.width[0] = 5
    logits = torch.full((2, BLOCK, MASK + 3), -1e9)
    logits[..., delete] = 0.0
    logits[..., 4] = -1.0
    policy = Indel(MASK, delete, split, 1, 0.0, 0.0, 4, 60)
    after, _ = policy(logits, rows)
    assert (after.canvas[0, 1:5] == 4).all()
    assert (after.canvas[1, 1:] != delete).all() and after.canvas[1, 1:].tolist() != [4] * (
        BLOCK - 1
    )
