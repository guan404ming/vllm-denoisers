import json
from pathlib import Path

import pytest
import torch
from denoisers.canvas.transition import advance
from denoisers.models.fast_dllm.config import FastDLLMQwenConfig
from denoisers.policies import build
from denoisers.policies.fast_dllm import FastDLLM
from support import make_rows

MASK = 15
VOCAB = 16
BLOCK = 32
SUB = 8
CONFIG = {
    "architectures": ["Fast_dLLM_QwenForCausalLM"],
    "model_type": "Fast_dLLM_Qwen",
    "bd_size": 32,
    "hidden_size": 1536,
    "num_hidden_layers": 28,
    "num_attention_heads": 12,
    "num_key_value_heads": 2,
    "vocab_size": 151936,
    "use_sliding_window": False,
}


def hf_step(x_t, logits, threshold):
    x_t = x_t.clone()
    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    mask_idx = x_t == MASK
    for sub in range(BLOCK // SUB):
        start, end = sub * SUB, (sub + 1) * SUB
        if mask_idx[:, start:end].sum() == 0:
            continue
        p = torch.softmax(logits[:, start:end].float(), dim=-1)
        x_1 = p.argmax(dim=-1)
        x1_p = p.gather(-1, x_1.unsqueeze(-1)).squeeze(-1)
        x1_p = torch.where(mask_idx[:, start:end], x1_p, -torch.inf)
        unmask = x1_p > threshold
        unmask[torch.arange(x_1.shape[0]), x1_p.argmax(dim=-1)] = True
        unmask = unmask & mask_idx[:, start:end]
        block = x_t[:, start:end]
        block[unmask] = x_1[unmask]
        break
    return x_t


@pytest.mark.parametrize("threshold", [0.0, 0.5, 0.9, 1.0])
def test_step_matches_hf_sub_block_loop(threshold):
    gen = torch.Generator().manual_seed(int(threshold * 10))
    canvas = torch.randint(0, MASK, (1, BLOCK), generator=gen)
    canvas[torch.rand(1, BLOCK, generator=gen) < 0.5] = MASK
    canvas[0, :3] = MASK
    logits = torch.randn(1, BLOCK, VOCAB, generator=gen) * 4
    logits[..., MASK] = -1e9
    policy = FastDLLM(MASK, BLOCK, SUB, threshold)
    after, done = policy(logits, make_rows(canvas, mask_id=MASK))
    assert torch.equal(after.canvas, hf_step(canvas, logits, threshold))
    assert not done.item()


def test_rows_advance_independently_and_finish():
    canvas = torch.full((2, BLOCK), MASK)
    canvas[0, :-1] = 1
    canvas[1, :SUB] = 1
    logits = torch.zeros(2, BLOCK, VOCAB)
    logits[..., 3] = 50.0
    policy = FastDLLM(MASK, BLOCK, SUB, 0.9)
    after, done = policy(logits, make_rows(canvas, mask_id=MASK))
    assert done.tolist() == [True, False]
    assert after.canvas[0, -1].item() == 3
    assert (after.canvas[1, SUB : 2 * SUB] == 3).all()
    assert (after.canvas[1, 2 * SUB :] == MASK).all()


def test_commit_seeds_next_block_with_last_logit():
    canvas = torch.full((1, BLOCK), 1)
    rows = make_rows(canvas, mask_id=MASK)
    rows.committing[:] = True
    logits = torch.zeros(1, BLOCK, VOCAB)
    logits[0, -1, 7] = 50.0
    logits[0, :-1, 2] = 50.0
    after, _sampled, num_sampled = advance(rows, logits, FastDLLM(MASK, BLOCK, SUB, 0.9), MASK)
    assert num_sampled.item() == BLOCK
    assert after.canvas[0, 0].item() == 7
    assert (after.canvas[0, 1:] == MASK).all()
    assert after.original[0].all()


def test_checkpoint_config_gets_decode_defaults(tmp_path: Path):
    (tmp_path / "config.json").write_text(json.dumps(CONFIG))
    cfg = FastDLLMQwenConfig.from_pretrained(tmp_path)
    assert cfg.canvas_length == cfg.bd_size == 32
    assert cfg.mask_token_id == 151665
    assert cfg.num_key_value_heads == 2
    policy = build(cfg.decode_policy, cfg)
    assert isinstance(policy, FastDLLM)
    assert policy.threshold == 0.9
