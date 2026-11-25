from types import SimpleNamespace

import pytest
import torch

from denoisers.models.llada2.routing import BlockRouter

EXPERTS = 16
CAPACITY = 6
TOPK = 3
BLOCK = 4
SCALE = 2.5


def hf_route(logits, bias, groups):
    scores = torch.sigmoid(logits.float())
    routing = scores + bias
    topk_idx = torch.empty(logits.shape[0], TOPK, dtype=torch.int64)
    for group in groups:
        block_scores = routing[group].max(dim=0).values
        allowed = torch.zeros(EXPERTS, dtype=torch.bool)
        allowed[block_scores.topk(CAPACITY).indices] = True
        masked = routing[group].masked_fill(~allowed, -torch.inf)
        topk_idx[group] = masked.topk(TOPK, dim=-1).indices
    weights = scores.gather(1, topk_idx)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return weights * SCALE, topk_idx


def make_router(gen):
    gate = SimpleNamespace(expert_bias=torch.randn(EXPERTS, generator=gen) * 0.1)
    return BlockRouter(gate, BLOCK, CAPACITY, SCALE), gate


@pytest.mark.parametrize("seed", range(5))
def test_aligned_blocks_match_hf(seed):
    gen = torch.Generator().manual_seed(seed)
    router, gate = make_router(gen)
    logits = torch.randn(3 * BLOCK, EXPERTS, generator=gen)
    router.set_blocks(torch.arange(3 * BLOCK) // BLOCK, 3)
    weights, ids = router(None, logits, TOPK, True)
    want_w, want_ids = hf_route(logits, gate.expert_bias, [range(4), range(4, 8), range(8, 12)])
    assert torch.equal(ids, want_ids)
    assert torch.allclose(weights, want_w)


def test_request_blocks_group_tokens_by_request_and_position():
    gen = torch.Generator().manual_seed(7)
    router, gate = make_router(gen)
    logits = torch.randn(10, EXPERTS, generator=gen)
    router.set_blocks(torch.tensor([0, 0, 0, 0, 1, 1, 9, 9, 9, 10]), 12)
    weights, ids = router(None, logits, TOPK, True)
    groups = [range(4), range(4, 6), range(6, 9), range(9, 10)]
    want_w, want_ids = hf_route(logits, gate.expert_bias, groups)
    assert torch.equal(ids, want_ids)
    assert torch.allclose(weights, want_w)


def test_block_ids_may_be_longer_than_the_batch_but_not_shorter():
    gen = torch.Generator().manual_seed(3)
    router, gate = make_router(gen)
    logits = torch.randn(2 * BLOCK, EXPERTS, generator=gen)
    with pytest.raises(RuntimeError):
        router(None, logits, TOPK, True)
    router.set_blocks(torch.arange(3 * BLOCK) // BLOCK, 3)
    _, ids = router(None, logits, TOPK, True)
    _, want_ids = hf_route(logits, gate.expert_bias, [range(4), range(4, 8)])
    assert torch.equal(ids, want_ids)
    router.set_blocks(torch.zeros(5, dtype=torch.int64), 1)
    with pytest.raises(RuntimeError):
        router(None, logits, TOPK, True)
