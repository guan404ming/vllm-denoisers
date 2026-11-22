import math

import pytest
import torch
from support import make_rows

from denoisers.canvas.transition import advance
from denoisers.ops.capped_stats import (
    capped_row_stats,
    capped_row_stats_reference,
    select_candidates,
    softcap,
)
from denoisers.ops.embedding import ExpectedEmbedding
from denoisers.policies.decode.entropy_bound import COEF, EntropyBound, remaining_steps

VOCAB = 16
BLOCK = 8
HIDDEN = 4
STEPS = 6


def make_policy(t_min=0.0, t_max=0.0, bound=0.1, confidence=0.5, tp_size=1, cap=None):
    gen = torch.Generator().manual_seed(0)
    return EntropyBound(
        vocab_size=VOCAB,
        max_steps=STEPS,
        t_min=t_min,
        t_max=t_max,
        entropy_bound=bound,
        confidence_threshold=confidence,
        embedding=ExpectedEmbedding(
            torch.randn(VOCAB, HIDDEN, generator=gen), tp_size=tp_size, scale=torch.tensor(2.0)
        ),
        softcap=cap,
    )


def rows_for(policy, n=2):
    canvas = policy.init_canvas(torch.zeros(n, BLOCK, dtype=torch.int64))
    rows = make_rows(canvas)
    rows.history = torch.zeros(n, 2, BLOCK, dtype=torch.int64)
    rows.soft = torch.zeros(n, BLOCK, HIDDEN)
    return rows


def peaked_logits(tokens: torch.Tensor, scale: float = 50.0) -> torch.Tensor:
    return torch.nn.functional.one_hot(tokens, VOCAB).float() * scale


def test_sparse_soft_embeds_match_exact_on_peaked_rows():
    policy = make_policy()
    probs = torch.zeros(3, VOCAB)
    probs[0, 3] = 1.0
    probs[1, [1, 5]] = 0.5
    probs[2, [2, 7, 9]] = torch.tensor([0.6, 0.3, 0.1])
    embedding = policy.embedding
    exact = embedding(probs)
    assert torch.allclose(embedding(select_candidates(probs)), exact, atol=1e-5)
    embedding.vocab_start, embedding.vocab_end = 4, 8
    partial = probs[:, 4:8] @ embedding.weight[:4] * 2.0
    assert torch.allclose(embedding(select_candidates(probs)), partial, atol=1e-5)


def test_remaining_steps_follows_the_fitted_features():
    mean, median, low, top, per_step, bias = COEF
    entropy = torch.tensor([[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])
    got = remaining_steps(entropy, torch.tensor([0, 3]))
    want0 = 0.5 * mean + 0.5 * low + 1.0 * top + bias
    want1 = 2.0 * mean + 2.0 * median + 2.0 * top + 3 * per_step + bias
    assert math.isclose(got[0].item(), want0, rel_tol=1e-4)
    assert math.isclose(got[1].item(), want1, rel_tol=1e-4)
    assert remaining_steps(torch.zeros(1, 4), torch.tensor([0])).item() >= 0.5


def test_converges_when_argmax_is_stable_and_confident():
    policy = make_policy(bound=0.1, confidence=0.5)
    rows = rows_for(policy, n=1)
    target = torch.arange(BLOCK)
    logits = peaked_logits(target).unsqueeze(0)
    rows, done = policy(logits, rows)
    assert not done.item()
    rows, done = policy(logits, rows)
    assert done.item()
    assert torch.equal(rows.canvas[0], target)
    assert torch.equal(rows.soft[0], torch.zeros(BLOCK, HIDDEN))


def test_commit_step_reinits_canvas_and_emits_argmax():
    policy = make_policy(bound=0.1, confidence=0.5)
    rows = rows_for(policy, n=1)
    target = torch.arange(BLOCK)
    logits = peaked_logits(target).unsqueeze(0)
    rows, _, _ = advance(rows, logits, policy, -1)
    rows, _, num = advance(rows, logits, policy, -1)
    assert rows.committing.item() and num.item() == 0
    rows, sampled, num = advance(rows, None, policy, -1)
    assert num.item() == BLOCK
    assert torch.equal(sampled[0], target)
    assert not rows.committing.item() and rows.step.item() == 0
    assert (rows.canvas != -1).all() and rows.history_len.item() == 0
    assert torch.equal(rows.soft[0], torch.zeros(BLOCK, HIDDEN))


class Params:
    def __init__(self, **extra):
        self.extra_args = extra
        self.logprob_token_ids = extra.pop("ids", None)


class State:
    def __init__(self, rows=4):
        self.canvas = torch.zeros(rows, BLOCK, dtype=torch.int64)


def bound_policy(**extra):
    policy = make_policy(bound=0.1, confidence=0.5)
    policy.bind(State())
    policy.add_request(1, Params(**extra))
    return policy


def test_seed_and_pins_hold_through_denoise_and_read_only_emits():
    seed = list(range(BLOCK))
    policy = bound_policy(
        diffusion_seed_canvas=seed, diffusion_pinned=[0, 3], diffusion_read_only=True
    )
    rows = rows_for(policy, n=1)
    rows.slots = torch.tensor([1])
    canvas = policy.init_canvas(rows.canvas, rows.slots, first=True)
    assert torch.equal(canvas[0], torch.tensor(seed))
    rows.canvas = canvas
    logits = peaked_logits(torch.full((BLOCK,), 9))
    rows, sampled, num = advance(rows, logits.unsqueeze(0), policy, -1)
    assert num.item() == 0 and rows.canvas[0, 0].item() == 0 and rows.canvas[0, 3].item() == 3
    assert rows.canvas[0, 1].item() == 9
    rows, sampled, num = advance(rows, logits.unsqueeze(0), policy, -1)
    assert num.item() == BLOCK and not rows.committing.item()
    assert sampled[0, 0].item() == 0 and sampled[0, 3].item() == 3 and sampled[0, 1].item() == 9


def test_constrained_read_and_step_cap():
    policy = bound_policy(diffusion_constrained=True, diffusion_max_steps=1, ids=[2, 5])
    rows = rows_for(policy, n=1)
    rows.slots = torch.tensor([1])
    logits = peaked_logits(torch.full((BLOCK,), 9)).unsqueeze(0)
    logits[0, :, 5] = 40.0
    after, done = policy(logits, rows)
    assert done.item()
    assert torch.equal(after.canvas[0], torch.full((BLOCK,), 5))


@pytest.mark.parametrize("seed", [0, 1])
def test_matches_vllm_sample_step(seed):
    dg = pytest.importorskip("vllm.model_executor.models.diffusion_gemma")
    torch._dynamo.config.disable = True
    n = 2
    policy = make_policy(bound=0.3, confidence=0.6, cap=3.0)
    states = dg.DiffusionGemmaRequestStates(
        max_num_reqs=n,
        canvas_length=BLOCK,
        vocab_size=VOCAB,
        max_denoising_steps=STEPS,
        device=torch.device("cpu"),
        hidden_size=HIDDEN,
        stability_threshold=2,
    )
    slots = torch.arange(n)
    for slot in range(n):
        states.add_request(slot)
    states.is_encoder_phase[:] = False
    rows = rows_for(policy, n)
    rows.canvas = states.canvas.clone()
    sampled = torch.zeros(n, BLOCK, dtype=torch.int32)
    num_sampled = torch.zeros(n, dtype=torch.int32)
    draft = torch.zeros(n, BLOCK, dtype=torch.int64)
    gen = torch.Generator().manual_seed(seed)
    for step in range(3 * STEPS):
        logits = torch.randn(n * BLOCK, VOCAB, generator=gen) * 3
        if step % 4 == 1:
            logits[:BLOCK] = peaked_logits(torch.arange(BLOCK))
        committing = states.is_encoder_phase.clone()
        temps = torch.zeros(n)
        torch.manual_seed(seed * 100 + step)
        argmax, sample, entropy, probs = dg.sample_row_stats_reference(
            softcap(logits, 3.0), temps, BLOCK, torch.float32
        )
        dg._compiled_sample_step(
            sample.view(n, BLOCK),
            argmax.view(n, BLOCK),
            entropy.view(n, BLOCK),
            probs.view(n, BLOCK, VOCAB),
            slots,
            slots,
            slots,
            torch.full((n,), BLOCK),
            states.canvas,
            states.argmax_canvas,
            states.step,
            states.is_encoder_phase,
            states.confident,
            states.self_conditioning_embeds,
            policy.embedding.weight,
            policy.embedding.scale,
            states.accepted_canvas_history,
            states.accepted_canvas_history_len,
            states.max_steps,
            states.pin_mask,
            states.seed_canvas,
            states.read_only,
            sampled,
            num_sampled,
            draft,
            confidence_threshold=policy.confidence_threshold,
            vocab_size=VOCAB,
            CL=BLOCK,
            ST=2,
            entropy_bound=policy.entropy_bound,
            sc_vocab_start=0,
            sc_vocab_end=VOCAB,
            tp_size=1,
            tp_group_name="",
        )
        torch.manual_seed(seed * 100 + step)
        rows, ours_sampled, ours_num = advance(rows, logits.view(n, BLOCK, VOCAB), policy, -1)
        assert torch.equal(rows.committing, states.is_encoder_phase)
        assert torch.equal(ours_num, num_sampled.long())
        assert torch.equal(ours_sampled[committing], sampled.long()[committing])
        assert torch.equal(rows.canvas[~committing], states.canvas[~committing])
        assert torch.allclose(rows.soft, states.self_conditioning_embeds, atol=1e-5)
        assert torch.equal(rows.history_len, states.accepted_canvas_history_len.long())
        rows.canvas[committing] = states.canvas[committing]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("cap", [None, 30.0])
@pytest.mark.parametrize("sparse", [False, True])
def test_capped_kernel_matches_reference(cap, sparse):
    gen = torch.Generator().manual_seed(0)
    logits = (torch.randn(6, 5000, generator=gen) * 40).cuda()
    logits[0, :10] = -torch.inf
    temps = torch.tensor([0.0, 0.5, 0.8]).cuda()
    got = capped_row_stats(logits, temps, 2, cap, torch.bfloat16, sparse)
    ref = capped_row_stats_reference(logits, temps, 2, cap, torch.float32, sparse)
    assert torch.equal(got[0], ref[0])
    assert torch.allclose(got[2], ref[2], atol=1e-3)
    assert torch.equal(got[1][:2], ref[1][:2])
    if sparse:
        assert torch.equal(got[3][0], ref[3][0])
        assert torch.allclose(got[3][1], ref[3][1], atol=1e-3)
    else:
        assert torch.allclose(got[3].float(), ref[3], atol=1e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_kernel_samples_follow_the_distribution():
    logits = torch.full((1, 5000), -torch.inf, device="cuda")
    logits[0, 7] = 2.0
    logits[0, 4100] = 2.0 + math.log(3.0)
    temps = torch.ones(1, device="cuda")
    draws = torch.stack(
        [capped_row_stats(logits, temps, 1, None, torch.float32)[1] for _ in range(400)]
    )
    share = (draws == 4100).float().mean().item()
    assert 0.65 < share < 0.85
    assert ((draws == 7) | (draws == 4100)).all()
