import pytest
import torch
from support import make_rows

from denoisers.canvas.transition import advance
from denoisers.ops.sampling import transfer_schedule
from denoisers.policies.decode.indel import Indel, apply_edits

MASK, DELETE, SPLIT = 20, 21, 22
VOCAB = 23
BLOCK = 8
STEPS = 8
MAX_POST = 3
MAX_STEPS = 60


def hf_apply_edits(block_tokens, old_block_tokens, snapshot):
    tokens, tracking = [], []
    for i, token in enumerate(block_tokens):
        if token == DELETE:
            continue
        if token == SPLIT:
            tokens.extend([MASK, old_block_tokens[i]])
            tracking.extend([False, snapshot[i]])
        else:
            tokens.append(token)
            tracking.append(snapshot[i])
    tokens, tracking = tokens[:BLOCK], tracking[:BLOCK]
    pad = BLOCK - len(tokens)
    return tokens + [MASK] * pad, tracking + [False] * pad


def greedy(logits, banned):
    logits = logits.clone()
    logits[list(banned)] = -torch.inf
    return int(logits.argmax())


def hf_decode_block(x, shift, logits_seq, threshold, editing_threshold):
    prompt = [i < shift for i in range(BLOCK)]
    is_original = [t == MASK for t in x]
    schedule = transfer_schedule(sum(is_original), STEPS).tolist()
    seen, post, step_id = set(), 0, 0
    while True:
        old = list(x)
        mask_index = [x[i] == MASK and not prompt[i] for i in range(BLOCK)]
        original_count = sum(1 for i in range(BLOCK) if is_original[i] and x[i] == MASK)
        new_count = sum(mask_index) - original_count
        post = post + 1 if original_count == 0 else 0
        if sum(mask_index) == 0 and post > MAX_POST:
            break
        if step_id >= MAX_STEPS:
            break
        logits = logits_seq[step_id]
        probs = torch.softmax(logits, dim=-1)
        x0 = [greedy(logits[i], {MASK}) for i in range(BLOCK)]
        p0 = [probs[i, x0[i]].item() for i in range(BLOCK)]
        mt2 = [False] * BLOCK
        if any(mask_index):
            if step_id < len(schedule):
                need = schedule[step_id] + new_count
                conf = [p0[i] if mask_index[i] else -float("inf") for i in range(BLOCK)]
                high = [conf[i] > threshold and mask_index[i] for i in range(BLOCK)]
                if sum(high) >= need:
                    mt2 = high
                else:
                    k = min(need, sum(mask_index))
                    for i in sorted(range(BLOCK), key=lambda i: -conf[i])[:k]:
                        mt2[i] = True
            else:
                mt2 = list(mask_index)
        editable = [not mask_index[i] and not prompt[i] for i in range(BLOCK)]
        t2t = [editable[i] and p0[i] > editing_threshold and x0[i] != old[i] for i in range(BLOCK)]
        final_round = post >= MAX_POST
        if final_round:
            mt2 = list(mask_index)
            for i in range(BLOCK):
                if (mt2[i] or t2t[i]) and x0[i] in (SPLIT, DELETE):
                    x0[i] = greedy(logits[i], {MASK, SPLIT, DELETE})
        blk = list(old)
        for i in range(BLOCK):
            if mt2[i] or t2t[i]:
                blk[i] = x0[i]
        if not final_round and tuple(blk) in seen and blk != old:
            changed = sorted((i for i in range(BLOCK) if blk[i] != old[i]), key=lambda i: p0[i])
            for i in changed[:5]:
                blk[i] = greedy(logits[i], {blk[i], MASK})
                if tuple(blk) not in seen:
                    break
        seen.add(tuple(blk))
        x, is_original = hf_apply_edits(blk, old, list(is_original))
        step_id += 1
        if x == old and MASK not in x:
            break
    return x, step_id


def random_logits(gen, n):
    logits = torch.randn(n, BLOCK, VOCAB, generator=gen) * 3
    logits[..., DELETE] += torch.where(torch.rand(n, BLOCK, generator=gen) < 0.08, 6.0, -4.0)
    logits[..., SPLIT] += torch.where(torch.rand(n, BLOCK, generator=gen) < 0.08, 6.0, -4.0)
    return logits


@pytest.mark.parametrize("seed", range(12))
def test_apply_edits_matches_hf(seed):
    gen = torch.Generator().manual_seed(seed)
    block = torch.randint(0, VOCAB, (6, BLOCK), generator=gen)
    old = torch.randint(0, MASK, (6, BLOCK), generator=gen)
    original = torch.rand(6, BLOCK, generator=gen) < 0.5
    out, out_original = apply_edits(block, old, original, MASK, DELETE, SPLIT)
    for r in range(6):
        tokens, tracking = hf_apply_edits(block[r].tolist(), old[r].tolist(), original[r].tolist())
        assert out[r].tolist() == tokens
        assert out_original[r].tolist() == tracking


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("threshold,editing_threshold", [(0.5, 0.0), (0.9, 0.5)])
def test_block_decode_matches_hf(seed, threshold, editing_threshold):
    gen = torch.Generator().manual_seed(seed)
    shift = seed % BLOCK
    policy = Indel(MASK, DELETE, SPLIT, STEPS, threshold, editing_threshold, MAX_POST, MAX_STEPS)
    tail = torch.randint(0, MASK, (shift,), generator=gen)
    canvas = torch.full((1, BLOCK), MASK)
    canvas[0, :shift] = tail
    logits_seq = random_logits(gen, MAX_STEPS + 1)

    expected, forwards = hf_decode_block(
        canvas[0].tolist(), shift, logits_seq, threshold, editing_threshold
    )

    rows = make_rows(canvas.clone(), shift=[shift], mask_id=MASK)
    for t in range(MAX_STEPS + 1):
        final = rows.canvas.clone()
        rows, _, num_sampled = advance(rows, logits_seq[t : t + 1], policy, MASK)
        if rows.committing.item():
            assert t + 1 == forwards
            assert rows.canvas[0].tolist() == expected
            break
        if num_sampled.item():
            assert final[0].tolist() == expected
            break
    else:
        raise AssertionError("block never committed")


def test_loop_escape_replaces_lowest_confidence_change():
    policy = Indel(MASK, DELETE, SPLIT, STEPS, 0.5, 0.0, MAX_POST, MAX_STEPS)
    before = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    written = torch.tensor([[1, 9, 3, 10, 5, 6, 7, 8]])
    logits = torch.full((1, BLOCK, VOCAB), -5.0)
    logits[0, 1, 9], logits[0, 1, 11] = 2.0, 1.0
    logits[0, 3, 10], logits[0, 3, 12] = 3.0, 1.0
    p0 = torch.softmax(logits, dim=-1).max(dim=-1).values
    rows = make_rows(before, mask_id=MASK)
    rows.history[0, 0] = written[0]
    rows.history_len[0] = 1
    out, history, history_len = policy._escape_loops(
        written, before, logits, p0, rows, torch.tensor([True])
    )
    assert out[0].tolist() == [1, 11, 3, 10, 5, 6, 7, 8]
    assert history[0, 1].tolist() == out[0].tolist()
    assert history_len.item() == 2
