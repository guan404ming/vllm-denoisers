from types import SimpleNamespace

import numpy as np
import torch

from denoisers.canvas.state import CanvasState
from denoisers.policies.decode import Dream, JointThreshold, LowConfidence
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
