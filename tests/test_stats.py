import math

import pytest
import torch

from denoisers.ops.stats import (
    _nucleus_reference,
    _reference,
    nucleus_bag,
    row_stats,
    sample_stats,
)
from denoisers.scheduler import probe

MASK, DELETE, SPLIT = 9, 7, 8


def _logits(device="cpu", dtype=torch.float32):
    torch.manual_seed(0)
    logits = torch.randn(5, 3, 16, device=device) * 4
    logits[0, 0, MASK] = 50.0
    logits[0, 1, DELETE] = 50.0
    logits[0, 1, SPLIT] = 49.0
    logits[1, 0, 2] = 30.0
    logits[1, 0, 11] = 30.0
    logits[2, 2] = -torch.inf
    logits[2, 2, 4] = 0.0
    return logits.to(dtype)


def test_reference_matches_torch_formulas():
    logits = _logits()
    stats = row_stats(logits, MASK, (DELETE, SPLIT))
    probs = torch.softmax(logits, dim=-1)
    candidates = probs.clone()
    candidates[..., MASK] = 0
    assert torch.equal(stats.x0, candidates.argmax(dim=-1))
    assert torch.allclose(stats.p0, probs.gather(-1, stats.x0.unsqueeze(-1)).squeeze(-1))
    assert torch.allclose(stats.entropy, probe.entropy(logits), atol=1e-5)
    assert stats.x0[0, 1] == DELETE and stats.alt[0, 1] not in (MASK, DELETE, SPLIT)
    assert stats.x0[1, 0] == 2
    assert stats.x0[2, 2] == 4 and stats.p0[2, 2] == 1.0 and stats.entropy[2, 2] == 0.0
    assert stats.x0.shape == (5, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_kernel_matches_reference(dtype):
    logits = _logits("cuda", dtype)
    wide = torch.randn(4, 2, 151936, device="cuda").to(dtype)
    for x in (logits, wide):
        active = torch.rand(x.shape[:-1], device="cuda") > 0.3
        got = row_stats(x, MASK, (DELETE, SPLIT), active)
        ref = _reference(
            x.reshape(-1, x.shape[-1]), MASK, (DELETE, SPLIT, MASK), active.reshape(-1)
        )
        assert torch.equal(got.x0.flatten(), ref.x0)
        assert torch.equal(got.alt.flatten(), ref.alt)
        assert torch.allclose(got.p0.flatten(), ref.p0, atol=1e-5)
        assert torch.allclose(got.entropy.flatten(), ref.entropy, atol=1e-4)


def test_inactive_rows_are_skipped():
    logits = _logits()
    active = torch.ones(5, 3, dtype=torch.bool)
    active[1] = False
    stats = row_stats(logits, MASK, (DELETE, SPLIT), active)
    full = row_stats(logits, MASK, (DELETE, SPLIT))
    assert torch.equal(stats.x0[0], full.x0[0]) and torch.equal(stats.p0[2:], full.p0[2:])
    assert (stats.x0[1] == 0).all() and (stats.p0[1] == 0).all() and (stats.entropy[1] == 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_sample_kernel_returns_tempered_and_plain_probabilities():
    torch.manual_seed(0)
    logits = (torch.randn(4, 8, 126464, device="cuda") * 4).bfloat16()
    active = torch.rand(4, 8, device="cuda") > 0.3
    x0, p0, plain = sample_stats(logits, MASK, 0.2, active)
    assert (x0[~active] == 0).all() and (p0[~active] == 0).all() and (plain[~active] == 0).all()
    assert (x0[active] != MASK).all()
    x = logits.float()
    pick = x0.unsqueeze(-1)
    assert torch.allclose(
        p0[active], torch.softmax(x / 0.2, -1).gather(-1, pick).squeeze(-1)[active], atol=1e-4
    )
    assert torch.allclose(
        plain[active], torch.softmax(x, -1).gather(-1, pick).squeeze(-1)[active], atol=1e-5
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_sample_kernel_follows_the_distribution():
    logits = torch.full((2000, 8200), -torch.inf, device="cuda")
    logits[:, 7] = 2.0
    logits[:, 8000] = 2.0 + math.log(3.0)
    logits[:, MASK] = 50.0
    x0, _, _ = sample_stats(logits, MASK, 1.0)
    assert ((x0 == 7) | (x0 == 8000)).all()
    assert 0.7 < (x0 == 8000).float().mean().item() < 0.8


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_nucleus_kernel_matches_reference():
    torch.manual_seed(0)
    vocab = 152064
    logits = (torch.randn(6, 4, vocab, device="cuda") * 0.4).bfloat16()
    logits[0, 0, 5] = 9.0
    logits[0, 1, MASK] = 9.0
    active = torch.rand(6, 4, device="cuda") > 0.3
    active[0, :2] = True
    ids, probs = nucleus_bag(logits, MASK, 0.1, active)
    ref_ids, ref_probs = _nucleus_reference(
        logits.reshape(-1, vocab), MASK, 0.1, active.reshape(-1)
    )
    got = torch.zeros(24, vocab, device="cuda").scatter_add_(
        1, ids.reshape(24, -1), probs.reshape(24, -1)
    )
    ref = torch.zeros(24, vocab, device="cuda").scatter_add_(1, ref_ids, ref_probs)
    assert torch.equal(got > 0, ref > 0)
    assert torch.allclose(got, ref, atol=1e-5)
    assert (got > 0).sum(-1).max() > 1
    assert got[0].nonzero().flatten().tolist() == [5] and (got[:, MASK] == 0).all()
