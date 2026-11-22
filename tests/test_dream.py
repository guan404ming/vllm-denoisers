import json
from pathlib import Path

import pytest
import torch
from support import make_rows

from denoisers.models.dream.config import DreamConfig
from denoisers.ops.sampling import exclude_filtered_mask
from denoisers.policies.decode import registry
from denoisers.policies.decode.dream import (
    Dream,
    nucleus_stats,
)

MASK = 15
VOCAB = 16
BLOCK = 8
CONFIG = {
    "architectures": ["DreamModel"],
    "model_type": "Dream",
    "hidden_size": 3584,
    "num_hidden_layers": 28,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "vocab_size": 152064,
    "mask_token_id": 151666,
    "use_sliding_window": False,
}


def hf_generate(x, logits_per_step, steps, eps):
    x = x.clone()
    timesteps = torch.linspace(1, eps, steps + 1)
    for i in range(steps):
        mask_index = x == MASK
        logits = logits_per_step[i]
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        mask_logits = logits[mask_index]
        t, s = timesteps[i], timesteps[i + 1]
        probs = torch.softmax(mask_logits.float(), dim=-1)
        _, x0 = probs.max(dim=-1)
        confidence = torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        num_mask_token = mask_index.sum() / mask_index.shape[0]
        number_transfer = (
            int(num_mask_token * (1 - s / t)) if i < steps - 1 else int(num_mask_token)
        )
        full_confidence = torch.full_like(x, -torch.inf, dtype=torch.float32)
        full_confidence[mask_index] = confidence
        if number_transfer > 0:
            _, transfer_index = torch.topk(full_confidence, number_transfer)
            x_ = torch.full_like(x, MASK)
            x_[mask_index] = x0
            rows = torch.arange(x.size(0)).unsqueeze(1).expand_as(transfer_index)
            x[rows, transfer_index] = x_[rows, transfer_index]
    return x


@pytest.mark.parametrize("steps", [2, 5, 8])
def test_policy_matches_hf(steps):
    gen = torch.Generator().manual_seed(steps)
    policy = Dream(MASK, BLOCK, steps)
    canvas = torch.full((1, BLOCK), MASK)
    canvas[0, :2] = torch.tensor([3, 7])
    logits_per_step = [torch.randn(1, BLOCK, VOCAB, generator=gen) * 4 for _ in range(steps)]
    logits_per_step = [torch.where(torch.arange(VOCAB) == MASK, -1e4, x) for x in logits_per_step]
    expected = hf_generate(canvas, logits_per_step, steps, 1e-3)
    out = canvas
    for step in range(steps):
        rows, result = policy(logits_per_step[step], make_rows(out, step=[step]))
        out = rows.canvas
    assert result.done.all()
    assert torch.equal(out, expected)


def hf_neg_entropy(logits, temperature, top_p):
    logits = logits / temperature
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    remove = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) > top_p
    remove[..., 1:] = remove[..., :-1].clone()
    remove[..., 0] = False
    drop = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, sorted_indices, remove)
    probs = torch.softmax(logits.masked_fill(drop, torch.finfo(logits.dtype).min), dim=-1)
    return torch.sum(probs * torch.log(probs + 1e-10), dim=-1), ~drop


@pytest.mark.parametrize("temperature, top_p", [(0.1, 0.9), (1.0, 0.9)])
def test_nucleus_matches_hf_sampling(temperature, top_p):
    gen = torch.Generator().manual_seed(0)
    logits = torch.randn(4, BLOCK, VOCAB, generator=gen)
    logits = torch.where(torch.arange(VOCAB) == MASK, -1e4, logits)
    neg_entropy, support = hf_neg_entropy(logits, temperature, top_p)
    x0, entropy = nucleus_stats(logits, MASK, temperature, top_p)
    assert (support.sum(dim=-1) > 1).any()
    assert torch.allclose(entropy, -neg_entropy, atol=1e-5)
    assert support.gather(-1, x0.unsqueeze(-1)).all()


def test_sampled_policy_reads_only_rows_that_predict_masked_positions():
    gen = torch.Generator().manual_seed(0)
    policy = Dream(MASK, BLOCK, BLOCK, temperature=0.1, top_p=0.9)
    canvas = torch.full((1, BLOCK), MASK)
    canvas[0, :3] = torch.tensor([3, 7, 2])
    logits = torch.randn(1, BLOCK, VOCAB, generator=gen)
    logits[0, [0, 1, BLOCK - 1]] = torch.nan
    rows, result = policy(logits, make_rows(canvas, step=[4], shift=[3]))
    assert torch.isfinite(result.entropy[0, 3:]).all()
    assert torch.equal(rows.canvas[0, :3], canvas[0, :3])
    assert (rows.canvas[0, 3:] != MASK).all()


def test_dense_scores_keep_entropy_on_every_row():
    gen = torch.Generator().manual_seed(0)
    policy = Dream(MASK, BLOCK, BLOCK)
    canvas = torch.full((1, BLOCK), MASK)
    canvas[0, :3] = torch.tensor([3, 7, 2])
    logits = torch.randn(1, BLOCK, VOCAB, generator=gen)
    rows, result = policy(logits, make_rows(canvas, step=[0], shift=[3]))
    assert (result.entropy[0, 1:3] == 0).all()
    policy.dense_scores = True
    rows, result = policy(logits, make_rows(canvas, step=[0], shift=[3]))
    assert (result.entropy[0, 1:3] > 0).all()


def test_prompt_tail_shortens_the_schedule():
    gen = torch.Generator().manual_seed(0)
    shift = 3
    policy = Dream(MASK, BLOCK, BLOCK)
    canvas = torch.full((1, BLOCK), MASK)
    canvas[0, :shift] = torch.tensor([3, 7, 2])
    steps = BLOCK - shift
    logits_per_step = [torch.randn(1, BLOCK, VOCAB, generator=gen) * 4 for _ in range(steps)]
    logits_per_step = [torch.where(torch.arange(VOCAB) == MASK, -1e4, x) for x in logits_per_step]
    expected = hf_generate(canvas, logits_per_step, steps, 1e-3)
    out = canvas
    for step in range(steps):
        rows, result = policy(logits_per_step[step], make_rows(out, step=[step], shift=[shift]))
        out = rows.canvas
    assert result.done.all()
    assert torch.equal(out, expected)


def test_checkpoint_config_gets_canvas_defaults(tmp_path: Path):
    (tmp_path / "config.json").write_text(json.dumps(CONFIG))
    cfg = DreamConfig.from_pretrained(tmp_path)
    assert cfg.canvas_length == 256
    assert cfg.mask_token_id == 151666
    assert cfg.decode_policy == "dream"
    assert cfg.denoise_steps is None
    assert cfg.vocab_size == 152064 and cfg.num_key_value_heads == 4
    cfg.denoise_steps = cfg.canvas_length
    policy = registry.resolve(cfg.decode_policy).from_config(cfg, None)
    assert policy.table.shape == (257, 256)
    assert policy.last[256] == 255 and policy.last[172] == 171


def test_top_k_keeps_a_dream_candidate_when_mask_ranks_first():
    logits = torch.tensor([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
    exclude_filtered_mask(logits, 2, torch.tensor([1, 3]), None)
    assert logits[1, 2] == 2
    filtered = logits[:1].masked_fill(logits[:1] < logits[0].max(), -torch.inf)
    token, entropy = nucleus_stats(filtered, 2, 1.0, 1.0)
    assert token.item() == 1 and entropy.item() == 0
