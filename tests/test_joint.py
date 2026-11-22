from types import SimpleNamespace

import pytest
import torch
from support import make_rows

from denoisers.policies.decode.joint import JointThreshold

MASK = 9
BLOCK = 8
MAX_POST = 3


def test_explicit_zero_thresholds_survive():
    cfg = SimpleNamespace(
        mask_token_id=MASK,
        temperature=0.0,
        confidence_threshold=0.5,
        editing_threshold=0.0,
        max_post_steps=16,
    )
    policy = JointThreshold.from_config(cfg, None)
    assert policy.threshold == 0.5 and policy.editing_threshold == 0.0


def hf_step(cur_block, prompt_len_in_block, x0, x0_p, threshold, editing_threshold):
    old = cur_block.clone()
    cur_block = cur_block.clone()
    active = cur_block == MASK
    mask_transfer = torch.zeros_like(x0, dtype=torch.bool)
    if active.sum() > 0:
        confidence = torch.where(active, x0_p, -torch.inf)
        high = (confidence > threshold) & active
        if high.sum().item() >= 1:
            mask_transfer = high
        else:
            _, idx = torch.topk(confidence, k=1)
            mask_transfer[idx] = True
    prompt_mask = torch.zeros(BLOCK, dtype=torch.bool)
    prompt_mask[:prompt_len_in_block] = True
    editable = ~active & ~prompt_mask
    editing_confidence = torch.where(editable, x0_p, -torch.inf)
    editing = (editing_confidence > editing_threshold) & editable & (x0 != old)
    final = mask_transfer | editing
    cur_block[final] = x0[final]
    return cur_block, editing.any().item()


@pytest.mark.parametrize("threshold", [0.5, 0.95])
@pytest.mark.parametrize("editing_threshold", [0.0, 0.5, 0.9])
def test_step_matches_hf(threshold, editing_threshold):
    gen = torch.Generator().manual_seed(int(threshold * 100 + editing_threshold * 10))
    policy = JointThreshold(MASK, threshold, editing_threshold, MAX_POST)
    rows = 24
    canvas = torch.randint(0, MASK, (rows, BLOCK), generator=gen)
    canvas[torch.rand(rows, BLOCK, generator=gen) < 0.4] = MASK
    canvas[:8] = torch.randint(0, MASK, (8, BLOCK), generator=gen)
    shift = torch.randint(0, BLOCK, (rows,), generator=gen)
    logits = torch.randn(rows, BLOCK, MASK + 1, generator=gen) * 4
    logits[..., MASK] = -torch.inf
    post_steps = torch.randint(0, MAX_POST, (rows,), generator=gen)
    probs = torch.softmax(logits, dim=-1)
    x0 = probs.argmax(dim=-1)
    p0 = probs.max(dim=-1).values

    after, done = policy(logits, make_rows(canvas, shift=shift, post_steps=post_steps))
    out = after.canvas

    for r in range(rows):
        expected, edited = hf_step(
            canvas[r], shift[r].item(), x0[r], p0[r], threshold, editing_threshold
        )
        assert torch.equal(out[r], expected)
        had_masks = (canvas[r] == MASK).any().item()
        settled = (not edited) or post_steps[r].item() + 1 >= MAX_POST
        assert done[r].item() == ((not had_masks) and settled)


def test_prompt_tail_is_never_edited():
    policy = JointThreshold(MASK, 0.5, 0.0, MAX_POST)
    canvas = torch.full((1, BLOCK), 3)
    logits = torch.zeros(1, BLOCK, MASK + 1)
    logits[..., 5] = 10.0
    out = policy(logits, make_rows(canvas, shift=[3]))[0].canvas
    assert out[0, :3].tolist() == [3, 3, 3]
    assert out[0, 3:].tolist() == [5] * (BLOCK - 3)


def test_post_steps_cap_ends_block():
    policy = JointThreshold(MASK, 0.5, 0.0, MAX_POST)
    canvas = torch.zeros(1, BLOCK, dtype=torch.int64)
    logits = torch.zeros(1, BLOCK, MASK + 1)
    logits[..., 1] = 10.0
    _, done = policy(logits, make_rows(canvas, post_steps=[0]))
    assert not done.item()
    _, done = policy(logits, make_rows(canvas, post_steps=[MAX_POST - 1]))
    assert done.item()


def test_no_edit_ends_block_immediately():
    policy = JointThreshold(MASK, 0.5, 0.0, MAX_POST)
    canvas = torch.ones(1, BLOCK, dtype=torch.int64)
    logits = torch.zeros(1, BLOCK, MASK + 1)
    logits[..., 1] = 10.0
    _, done = policy(logits, make_rows(canvas, post_steps=[0]))
    assert done.item()
