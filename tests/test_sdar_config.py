import json
from pathlib import Path

from denoisers.models.sdar.config import SDARConfig, SDARMoeConfig
from denoisers.policies import build

CONFIG = {
    "architectures": ["SDARForCausalLM"],
    "model_type": "sdar",
    "hidden_size": 2048,
    "num_hidden_layers": 28,
    "num_attention_heads": 16,
    "num_key_value_heads": 8,
    "vocab_size": 151936,
    "rope_theta": 1000000,
    "tie_word_embeddings": False,
}
MOE_CONFIG = {
    **CONFIG,
    "architectures": ["SDARMoeForCausalLM"],
    "model_type": "sdar_moe",
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 768,
    "norm_topk_prob": True,
}


def test_checkpoint_config_gets_canvas_defaults(tmp_path: Path):
    (tmp_path / "config.json").write_text(json.dumps(CONFIG))
    cfg = SDARConfig.from_pretrained(tmp_path)
    assert cfg.canvas_length == 4
    assert cfg.denoise_steps is None
    assert cfg.mask_token_id == 151669
    assert cfg.decode_policy == "low_confidence"
    assert cfg.hidden_size == 2048 and cfg.num_key_value_heads == 8
    cfg.update({"canvas_length": 32})
    cfg.denoise_steps = cfg.canvas_length
    policy = build(cfg.decode_policy, cfg)
    assert policy.threshold == 0.85
    assert len(policy.schedule) == 32


def test_moe_config_keeps_expert_fields(tmp_path: Path):
    (tmp_path / "config.json").write_text(json.dumps(MOE_CONFIG))
    cfg = SDARMoeConfig.from_pretrained(tmp_path)
    assert cfg.canvas_length == 4
    assert cfg.num_experts == 128 and cfg.num_experts_per_tok == 8
    assert cfg.norm_topk_prob is True
