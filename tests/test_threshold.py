import pytest
import torch
from support import make_rows

from denoisers.ops.sampling import sample, transfer_schedule, unmask
from denoisers.policies.decode.low_confidence import LowConfidence

MASK = 9
BLOCK = 8


def hf_step(cur_block, x0, x0_p, num_to_transfer, threshold):
    cur_block = cur_block.clone()
    active_block_mask = cur_block == MASK
    transfer_index = torch.zeros_like(x0, dtype=torch.bool)
    confidence = torch.where(active_block_mask, x0_p, -torch.inf)
    high_conf_mask = confidence > threshold
    if high_conf_mask.sum().item() >= num_to_transfer:
        transfer_index = high_conf_mask
    else:
        _, idx = torch.topk(confidence, k=min(num_to_transfer, active_block_mask.sum().item()))
        transfer_index[idx] = True
    cur_block[transfer_index] = x0[transfer_index]
    return cur_block


@pytest.mark.parametrize("steps", [1, 3, 8])
@pytest.mark.parametrize("threshold", [0.0, 0.5, 0.95])
def test_unmask_matches_hf(steps, threshold):
    gen = torch.Generator().manual_seed(steps * 100 + int(threshold * 100))
    schedule = transfer_schedule(BLOCK, steps)
    rows = 16
    canvas = torch.randint(0, MASK, (rows, BLOCK), generator=gen)
    canvas[torch.rand(rows, BLOCK, generator=gen) < 0.6] = MASK
    x0 = torch.randint(0, MASK, (rows, BLOCK), generator=gen)
    p0 = torch.rand(rows, BLOCK, generator=gen)
    step = torch.randint(0, steps, (rows,), generator=gen)

    out = unmask(canvas, x0, p0, schedule[step], threshold, MASK)

    for r in range(rows):
        expected = hf_step(canvas[r], x0[r], p0[r], schedule[step[r]].item(), threshold)
        assert torch.equal(out[r], expected)


def test_full_block_decodes_in_steps():
    gen = torch.Generator().manual_seed(0)
    policy = LowConfidence(MASK, BLOCK, BLOCK, 0.99)
    canvas = torch.full((4, BLOCK), MASK)
    canvas[0, :3] = 1
    for step in range(BLOCK):
        logits = torch.randn(4, BLOCK, MASK + 1, generator=gen)
        rows, result = policy(logits, make_rows(canvas, step=[step] * 4))
        canvas = rows.canvas
    assert result.done.all()
    assert torch.equal(canvas[0, :3], torch.ones(3, dtype=canvas.dtype))


def test_mask_token_is_never_chosen():
    logits = torch.zeros(2, BLOCK, MASK + 1)
    logits[..., MASK] = 10.0
    x0, p0 = sample(logits, MASK)
    assert (x0 != MASK).all()
    assert torch.allclose(p0, torch.softmax(logits, dim=-1)[..., 0])


def test_block_ends_after_last_step():
    policy = LowConfidence(MASK, BLOCK, 2, 0.99)
    canvas = torch.full((1, BLOCK), MASK)
    logits = torch.zeros(1, BLOCK, MASK + 1)
    _, result = policy(logits, make_rows(canvas, step=[0]))
    assert not result.done.item()
    _, result = policy(logits, make_rows(canvas, step=[1]))
    assert result.done.item()


def test_sample_with_temperature_skips_inactive_rows():
    gen = torch.Generator().manual_seed(0)
    logits = torch.randn(4, BLOCK, MASK + 1, generator=gen)
    active = torch.rand(4, BLOCK, generator=gen) < 0.5
    logits[~active] = torch.nan
    torch.manual_seed(0)
    x0, p0 = sample(logits, MASK, temperature=0.7, active=active)
    assert torch.isfinite(p0).all()
    assert (x0[~active] == 0).all() and (p0[~active] == 0).all()
    assert (x0[active] != MASK).all() and (p0[active] > 0).all()
    probs = torch.softmax(logits[active] / 0.7, dim=-1)
    assert torch.allclose(p0[active], probs.gather(-1, x0[active].unsqueeze(-1)).squeeze(-1))
