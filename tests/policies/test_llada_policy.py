import torch
from support import make_rows

from denoisers.policies.decode.llada import LLaDA

MASK = 15
VOCAB = 16
BLOCK = 8
SUB = 4


def reference(x, logits_per_step, steps, block_length):
    x = x.clone()
    gen_length = x.shape[1]
    num_blocks = gen_length // block_length
    steps = steps // num_blocks
    call = 0
    for num_block in range(num_blocks):
        lo, hi = num_block * block_length, (num_block + 1) * block_length
        mask_num = (x[:, lo:hi] == MASK).sum(dim=1, keepdim=True)
        base, remainder = mask_num // steps, mask_num % steps
        transfer = torch.zeros(x.shape[0], steps, dtype=torch.int64) + base
        for i in range(x.shape[0]):
            transfer[i, : remainder[i]] += 1
        for i in range(steps):
            mask_index = x == MASK
            logits = logits_per_step[call]
            call += 1
            p = torch.softmax(logits.float(), dim=-1)
            x0 = p.argmax(dim=-1)
            x0_p = p.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
            x0_p[:, hi:] = -float("inf")
            confidence = torch.where(mask_index, x0_p, -float("inf"))
            for j in range(x.shape[0]):
                _, idx = torch.topk(confidence[j], k=int(transfer[j, i]))
                x[j, idx] = x0[j, idx]
    return x


def test_llada_policy_matches_reference_loop():
    torch.manual_seed(0)
    steps = BLOCK
    logits_per_step = torch.randn(steps, 2, BLOCK, VOCAB)
    logits_per_step[..., MASK] = -1e9
    x = torch.full((2, BLOCK), MASK)
    expected = reference(x, logits_per_step, steps, SUB)
    policy = LLaDA(MASK, BLOCK, SUB, steps)
    rows = make_rows(x.clone(), mask_id=MASK)
    for i in range(steps):
        rows, result = policy(logits_per_step[i], rows)
        rows.step = rows.step + 1
    assert result.done.all()
    assert torch.equal(rows.canvas, expected)


def test_llada_policy_counts_sub_blocks_from_prompt_tail():
    torch.manual_seed(0)
    tail = 2
    logits_per_step = torch.randn(BLOCK, 2, tail + BLOCK, VOCAB)
    logits_per_step[..., MASK] = -1e9
    x = torch.full((2, BLOCK), MASK)
    expected = reference(x, logits_per_step[..., tail:, :], BLOCK, SUB)
    canvas = torch.cat([torch.zeros(2, tail, dtype=torch.int64), x], dim=1)
    policy = LLaDA(MASK, tail + BLOCK, SUB, BLOCK)
    rows = make_rows(canvas, shift=torch.full((2,), tail), mask_id=MASK)
    for i in range(BLOCK):
        rows, result = policy(logits_per_step[i], rows)
        rows.step = rows.step + 1
    assert result.done.all()
    assert torch.equal(rows.canvas[:, tail:], expected)


def test_llada_policy_ranks_sampled_tokens_by_untempered_confidence():
    torch.manual_seed(0)
    logits = torch.zeros(1, BLOCK, VOCAB)
    logits[..., MASK] = -1e9
    logits[0, 0, 3] = 3.0
    logits[0, 1, 3] = 10.0
    logits[0, 1, 4] = 9.5
    policy = LLaDA(MASK, BLOCK, SUB, BLOCK, temperature=0.05)
    rows, _ = policy(logits, make_rows(torch.full((1, BLOCK), MASK), mask_id=MASK))
    assert rows.canvas[0, 1] == 3
    assert (rows.canvas[0, [0, 2, 3]] == MASK).all()


def test_llada_finishes_partial_sub_block_before_commit():
    width = 6
    policy = LLaDA(MASK, width, SUB, width)
    logits = torch.zeros(1, width, VOCAB)
    rows = make_rows(torch.full((1, width), MASK))
    for step in range(policy.last + 1):
        rows.step = torch.tensor([step])
        rows, result = policy(logits, rows)
        if result.done.item():
            break
    assert result.done.item() and (rows.canvas != MASK).all()
