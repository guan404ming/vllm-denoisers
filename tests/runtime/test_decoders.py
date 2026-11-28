from types import SimpleNamespace

import numpy as np
import torch

from denoisers.canvas.state import CanvasState
from denoisers.policies.decode import Dream, Indel, JointThreshold, LowConfidence
from denoisers.runtime.decoders import Decoders

MASK = 9
BLOCK = 8


def make_decoders():
    built = {
        JointThreshold: JointThreshold(MASK, 0.9, 0.5, 3),
        Dream: Dream(MASK, BLOCK, BLOCK),
    }
    state = CanvasState(4, BLOCK, MASK, torch.device("cpu"))
    default = LowConfidence(MASK, BLOCK, BLOCK, 0.9)
    return Decoders(default, built.__getitem__, state, 4), default, built


def request(name):
    return SimpleNamespace(extra_args=None if name is None else {"decode_policy": name})


def test_requests_pick_registered_compatible_policies():
    decoders, default, built = make_decoders()
    assert decoders.assign(0, request(None)) is default
    assert decoders.assign(1, request("joint")) is built[JointThreshold]
    assert decoders.assign(2, request("dream")) is default
    assert decoders.assign(3, request("denoisers.policies.decode.joint.JointThreshold")) is default
    groups = decoders.groups(np.array([0, 1, 2, 3]))
    assert [(p, idx.tolist()) for p, idx in groups] == [
        (default, [0, 2, 3]),
        (built[JointThreshold], [1]),
    ]
    assert decoders.groups(np.array([0, 2]))[0] == (default, None)
    assert decoders.flags(np.array([0, 1]), "commit_logits").tolist() == [True, False]
    assert not decoders.all(np.array([0, 1]), "sparse_logits")
    assert decoders.assign(3, request("indel")) is default


def test_requests_keep_the_default_without_a_builder(caplog):
    state = CanvasState(4, BLOCK, MASK, torch.device("cpu"))
    default = LowConfidence(MASK, BLOCK, BLOCK, 0.9)
    decoders = Decoders(default, None, state, 4)
    assert decoders.assign(0, request("joint")) is default
    assert decoders.assign(1, request(None)) is default
    assert "Per-request decode_policy is off" in caplog.text


def test_request_history_is_lazy_and_preserves_active_slots_when_it_grows():
    decoders, default, built = make_decoders()
    state = decoders.state
    slots = torch.tensor([0])
    plain = decoders.assign(0, request(None))
    assert plain is default and state.gather(slots, plain.history_length).history is None
    assert state.history is None and state.history_len is None

    built[Indel] = Indel(MASK, 10, 11, 3, 0.5, 0.9, 2, 20, history_length=2)
    decoder = decoders.assign(0, request("indel"))
    rows = state.gather(slots, decoder.history_length)
    assert rows.history.shape == (1, 2, BLOCK)
    rows.history[:, 0] = 7
    rows.history_len.fill_(1)
    state.scatter(slots, rows)

    longer = state.gather(torch.tensor([1]), history_length=5)
    assert longer.history.shape == (1, 5, BLOCK)
    rows = state.gather(slots, decoder.history_length)
    assert rows.history_len.item() == 1 and (rows.history[:, 0] == 7).all()
    state.reset(0)
    plain = decoders.assign(0, request(None))
    assert state.gather(slots, plain.history_length).history is None
    assert state.gather(slots, decoder.history_length).history_len.item() == 0
