import numpy as np
import torch
from support import make_rows

from denoisers.canvas.state import CanvasState
from denoisers.policies.decode import LLaDA
from denoisers.policies.kv import Masked, Window
from denoisers.runtime.kv import KvRuntime, inverse_index, layout

MASK = 9
BLOCK = 8


def no(n=1):
    return torch.zeros(n, dtype=torch.bool)


def test_layout_keeps_masked_first_and_full_rows_identity():
    keep = torch.tensor(
        [
            [False, True, False, True, True, False, False, False],
            [True, True, True, True, True, True, True, True],
            [False, False, True, False, False, False, False, False],
        ]
    )
    full = torch.tensor([False, False, True])
    active, length = layout(keep, torch.full((3,), BLOCK), full, shifted=False)
    assert length.tolist() == [3, BLOCK, BLOCK]
    assert active[0, :3].tolist() == [1, 3, 4]
    assert sorted(active[0].tolist()) == list(range(BLOCK))
    assert active[1].tolist() == list(range(BLOCK))
    assert active[2].tolist() == list(range(BLOCK))


def test_layout_shifted_includes_left_neighbour():
    keep = torch.tensor([[False, False, True, False, False, True, False, False]])
    active, length = layout(keep, torch.tensor([BLOCK]), no(), True)
    assert length.tolist() == [4]
    assert active[0, :4].tolist() == [1, 2, 4, 5]


def test_layout_respects_truncated_width():
    keep = torch.tensor([[True, True, True, True, True, False, False, False]])
    _, length = layout(keep, torch.tensor([5]), no(), False)
    assert length.tolist() == [BLOCK]


def test_layout_without_kept_cells_is_full():
    keep = torch.zeros(1, BLOCK, dtype=torch.bool)
    active, length = layout(keep, torch.tensor([BLOCK]), no(), False)
    assert length.tolist() == [BLOCK]
    assert active[0].tolist() == list(range(BLOCK))


def test_layout_puts_prefix_rows_first_within_budget():
    keep = torch.tensor(
        [
            [False, True, False, True, False, False, False, False],
            [True, True, True, True, True, True, False, False],
        ]
    )
    active, length = layout(keep, torch.full((2,), BLOCK), no(2), False, torch.tensor([2, 5]))
    assert length.tolist() == [4, BLOCK]
    assert active[0, :4].tolist() == [-2, -1, 1, 3]
    assert active[1].tolist() == [-2, -1, 0, 1, 2, 3, 4, 5]


def test_inverse_index_skips_prefix_rows():
    active = torch.tensor(
        [[1, 3, 4, 0, 2, 5, 6, 7], list(range(BLOCK)), [-2, -1, 1, 3, 0, 2, 4, 5]]
    )
    inv = inverse_index(active, torch.tensor([3, BLOCK, 4]), torch.tensor([0, 3, 11]))
    assert inv[0].tolist() == [-1, 0, -1, 1, 2, -1, -1, -1]
    assert inv[1].tolist() == [3, 4, 5, 6, 7, 8, 9, 10]
    assert inv[2].tolist() == [-1, 13, -1, 14, -1, -1, -1, -1]
    assert inv.is_contiguous()


def llada():
    return LLaDA(MASK, BLOCK, sub_block=4, steps=8)


def test_window_keeps_current_sub_block_and_refreshes_on_move():
    first = torch.tensor([[1, MASK, MASK, 2, MASK, MASK, MASK, MASK]])
    later = torch.tensor([[1, 3, MASK, 2, MASK, MASK, MASK, MASK]])
    moved = torch.tensor([[1, 3, 4, 2, MASK, MASK, MASK, MASK]])
    window = Window(llada())
    keep = window.keep(make_rows(first), make_rows(later))
    assert keep[0].tolist() == [True] * 4 + [False] * 4
    assert window.keep(make_rows(later), make_rows(moved)).all()
    wide = Window(llada(), prefix_look=1, after_look=2)
    keep = wide.keep(make_rows(first), make_rows(later))
    assert keep[0].tolist() == [True] * 6 + [False] * 2
    assert wide.prefix(make_rows(first), make_rows(later)).tolist() == [1]
    assert wide.prefix(make_rows(moved), make_rows(moved)).tolist() == [0]


def test_window_without_focus_keeps_everything():
    canvas = torch.tensor([[1, MASK, 2, MASK, 3, 4, 5, 6]])
    window = Window(Masked(MASK))
    window.policy = type("NoFocus", (), {"focus": lambda self, rows: None})()
    assert window.keep(make_rows(canvas), make_rows(canvas)).all()


def runtime(build_policy, refresh_every=0):
    state = CanvasState(3, BLOCK, MASK, torch.device("cpu"))
    return KvRuntime(build_policy, state, shifted=False, refresh_every=refresh_every)


def rows_at(slots, canvas, step, start):
    rows = make_rows(canvas, step=step)
    rows.slots, rows.start = slots, torch.tensor(start)
    return rows


def test_runtime_refreshes_committed_prefix_with_history_tokens():
    kv = runtime(lambda decoder: Window(decoder, prefix_look=2))
    decoder = llada()
    policy = kv.for_decoder(decoder)
    slots = torch.tensor([2])
    first = torch.tensor([[1, MASK, MASK, 2, MASK, MASK, MASK, MASK]])
    later = torch.tensor([[1, 3, MASK, 2, MASK, MASK, MASK, MASK]])
    after = rows_at(slots, later, [2], [10])
    active = kv.update(policy, slots, rows_at(slots, first, [1], [10]), after, full=False)
    assert active[0, :6].tolist() == [-2, -1, 0, 1, 2, 3]
    other = LLaDA(MASK, BLOCK, sub_block=2, steps=8)
    other_policy = kv.for_decoder(other)
    other_slots = torch.tensor([0])
    other_rows = rows_at(other_slots, first, [1], [10])
    other_active = kv.update(other_policy, other_slots, other_rows, other_rows, full=False)
    assert other_active[0, :4].tolist() == [-2, -1, 0, 1]
    assert kv.for_decoder(decoder) is policy
    history = torch.arange(3 * 20).view(3, 20)
    tokens = kv.draft_tokens(policy, after, active, history)
    assert tokens[0, :6].tolist() == [48, 49, 1, 3, MASK, 2]
    kv.sync()
    assert kv.widths(np.array([2])).tolist() == [6]
    assert not kv.identity(np.array([2])).any()
    assert kv.pruned(slots, torch.tensor([6])).tolist() == [True]


def test_runtime_forces_full_rows_on_refresh_and_on_request():
    kv = runtime(lambda decoder: Masked(MASK), refresh_every=4)
    policy = kv.for_decoder(llada())
    slots = torch.tensor([0, 1])
    canvas = torch.tensor([[1, MASK, 2, 3, 4, 5, 6, 7]] * 2)
    before = rows_at(slots, canvas, [3, 2], [0, 0])
    after = rows_at(slots, canvas, [4, 3], [0, 0])
    kv.update(policy, slots, before, after, full=False)
    kv.sync()
    assert kv.widths(np.array([0, 1])).tolist() == [BLOCK, 1]
    assert kv.identity(np.array([0, 1])).tolist() == [True, False]
    kv.update(policy, slots, before, after, full=True)
    kv.sync()
    assert kv.identity(np.array([0, 1])).all()
