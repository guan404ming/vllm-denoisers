from types import SimpleNamespace

import torch
from support import make_rows

from denoisers.canvas.transition import advance
from denoisers.policies.decode import SigmaLatent

DIM = 2
VOCAB = 5
GRID = torch.tensor([6.0, 1.0, -3.0], dtype=torch.float64)


def make_policy():
    matrix = torch.randn(VOCAB, DIM, generator=torch.Generator().manual_seed(0))
    model = SimpleNamespace(gamma_grid=lambda calls: GRID[:calls], embedding_matrix=lambda: matrix)
    return SigmaLatent(model, 3, DIM), matrix


def latent_rows(step):
    rows = make_rows(torch.zeros(2, 4, dtype=torch.int64), step=step)
    rows.soft = torch.randn(2, 4, 2 * DIM + 1, generator=torch.Generator().manual_seed(1))
    return rows


def test_fresh_block_starts_from_noise_without_self_conditioning():
    policy, _ = make_policy()
    soft = policy.init_soft(torch.full((2, 4, 2 * DIM + 1), 7.0))
    assert soft[..., DIM : 2 * DIM].abs().sum() == 0
    assert soft[..., -1].eq(1).all() and soft[..., :DIM].std() > 0


def test_middle_step_carries_mean_and_next_step():
    policy, matrix = make_policy()
    rows = latent_rows([0, 2])
    logits = torch.randn(2, 4, VOCAB, generator=torch.Generator().manual_seed(2))
    after, result = policy(logits, rows)
    gamma = GRID[0].float()
    prior = torch.sigmoid(-gamma).sqrt() / torch.sigmoid(gamma) * rows.soft[0, :, :DIM]
    mean = (logits[0] + prior @ matrix.t()).softmax(-1) @ matrix
    assert result.done.tolist() == [False, True]
    assert torch.allclose(after.soft[0, :, DIM : 2 * DIM], mean, atol=1e-6)
    assert after.soft[0, :, -1].eq(2).all() and torch.equal(after.canvas[0], rows.canvas[0])


def test_last_step_commits_prior_argmax_and_clears_latents():
    policy, matrix = make_policy()
    rows = latent_rows([2, 2])
    logits = torch.randn(2, 4, VOCAB, generator=torch.Generator().manual_seed(3))
    after, result = policy(logits, rows)
    gamma = GRID[2].float()
    prior = torch.sigmoid(-gamma).sqrt() / torch.sigmoid(gamma) * rows.soft[..., :DIM]
    assert result.done.all() and after.soft.abs().sum() == 0
    assert torch.equal(after.canvas, (logits + prior @ matrix.t()).argmax(-1))


def test_committed_block_restarts_from_noise():
    policy, _ = make_policy()
    rows = latent_rows([0, 0])
    rows.committing = torch.tensor([True, False])
    after, _, _, _ = advance(rows, None, policy, -1)
    assert after.soft[0, :, -1].eq(1).all() and after.soft[0, :, DIM : 2 * DIM].abs().sum() == 0
    assert torch.equal(after.soft[1], rows.soft[1])
