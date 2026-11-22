import numpy as np
import pytest
import torch

from denoisers.canvas.prune import RowPruning
from denoisers.canvas.state import CanvasState
from denoisers.pruners import PRUNERS, MaskedRows, RowPruner, register_pruner
from support import make_rows

MASK = 9


def test_masked_rows_keeps_masked_cells_and_full_steps():
    canvas = torch.tensor([[1, MASK, 2, MASK], [MASK, 3, 4, 5], [6, 7, MASK, 8]])
    before = make_rows(canvas, step=[3, 3, 3], shift=[0, 0, 0])
    after = make_rows(canvas, step=[4, 0, 4], shift=[0, 0, 0])
    after.committing[2] = True
    keep = MaskedRows(MASK).keep(before, after)
    assert keep[0].tolist() == [False, True, False, True]
    assert keep[1].all() and keep[2].all()


def test_masked_rows_refresh_forces_full_rows():
    canvas = torch.tensor([[1, MASK, 2, 3]])
    before = make_rows(canvas, step=[3], shift=[0])
    after = make_rows(canvas, step=[4], shift=[0])
    assert not MaskedRows(MASK, refresh=3).keep(before, after).all()
    assert MaskedRows(MASK, refresh=2).keep(before, after).all()


def test_register_pruner():
    @register_pruner("test_masked")
    class Custom(MaskedRows):
        pass

    assert PRUNERS.pop("test_masked") is Custom
    with pytest.raises(ValueError):
        register_pruner("test_plain")(object)
    assert issubclass(MaskedRows, RowPruner)


def test_row_pruning_tracks_widths_per_slot():
    state = CanvasState(3, 4, MASK, torch.device("cpu"))
    pruning = RowPruning(MaskedRows(MASK), state, shifted=False)
    canvas = torch.tensor([[1, MASK, 2, 3], [MASK, MASK, 4, 5]])
    slots = torch.tensor([2, 0])
    before = make_rows(canvas, step=[3, 3], shift=[0, 0])
    after = make_rows(canvas, step=[4, 4], shift=[0, 0])
    active = pruning.update(slots, before, after, full=False)
    assert active[0, :1].tolist() == [1] and active[1, :2].tolist() == [0, 1]
    pruning.sync()
    assert pruning.widths(np.array([0, 1, 2])).tolist() == [2, 4, 1]
    assert pruning.unpruned(np.array([0, 1, 2])).tolist() == [False, True, False]
    pruning.update(slots, before, after, full=True)
    pruning.sync()
    assert pruning.widths(np.array([0, 2])).tolist() == [4, 4]
