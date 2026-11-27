import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from denoisers.models.a2d.config import A2DQwen2Config, A2DQwen3Config
from denoisers.policies.decode import registry

CONFIG = {
    "hidden_size": 1024,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "rope_theta": 1000000,
    "tie_word_embeddings": True,
}


def load(tmp_path: Path, cls, model_type: str):
    config = {"architectures": [cls.__name__], "model_type": model_type, **CONFIG}
    (tmp_path / "config.json").write_text(json.dumps(config))
    return cls.from_pretrained(tmp_path)


def verify(cfg, model: str, overrides=None):
    model_config = SimpleNamespace(model=model, hf_overrides=overrides or {})
    cfg.verify(SimpleNamespace(model_config=model_config))


@pytest.mark.parametrize(
    ("cls", "model_type", "mask"),
    [(A2DQwen3Config, "a2d-qwen3", 151669), (A2DQwen2Config, "a2d-qwen2", 151665)],
)
def test_bd3lm_checkpoint_unmasks_one_token_per_step(tmp_path: Path, cls, model_type, mask):
    cfg = load(tmp_path, cls, model_type)
    verify(cfg, "dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1")
    assert cfg.canvas_length == 32 and cfg.mask_token_id == mask
    assert cfg.decode_policy == "low_confidence" and cfg.tie_word_embeddings is True
    policy = registry.resolve(cfg.decode_policy).from_config(cfg, None)
    assert policy.threshold == 2.0
    assert policy.schedule.tolist() == [1] * 32


def test_mdlm_checkpoint_selects_llada_sub_blocks(tmp_path: Path):
    cfg = load(tmp_path, A2DQwen3Config, "a2d-qwen3")
    cfg.canvas_length = 128
    verify(cfg, "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1", {"canvas_length": 128})
    assert cfg.decode_policy == "llada" and cfg.sub_block_length == 64
    assert cfg.canvas_length == 128 and cfg.denoise_steps == 128
    policy = registry.resolve(cfg.decode_policy).from_config(cfg, None)
    assert policy.steps_per_block == 64 and policy.schedule.tolist() == [1] * 64
