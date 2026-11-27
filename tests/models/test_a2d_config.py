import json
from pathlib import Path

from denoisers.models.a2d.config import A2DQwen3Config
from denoisers.policies.decode import registry

CONFIG = {
    "architectures": ["A2DQwen3LMHeadModel"],
    "model_type": "a2d-qwen3",
    "hidden_size": 1024,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "vocab_size": 151936,
    "rope_theta": 1000000,
    "tie_word_embeddings": True,
}


def test_checkpoint_config_gets_bd3lm_defaults(tmp_path: Path):
    (tmp_path / "config.json").write_text(json.dumps(CONFIG))
    cfg = A2DQwen3Config.from_pretrained(tmp_path)
    assert cfg.canvas_length == 32 and cfg.mask_token_id == 151669
    assert cfg.decode_policy == "low_confidence" and cfg.tie_word_embeddings is True
    cfg.denoise_steps = cfg.canvas_length
    policy = registry.resolve(cfg.decode_policy).from_config(cfg, None)
    assert policy.threshold == 2.0
    assert policy.schedule.tolist() == [1] * 32
