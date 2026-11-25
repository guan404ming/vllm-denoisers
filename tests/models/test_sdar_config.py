import json
from pathlib import Path

from denoisers.models.sdar.config import SDARConfig
from denoisers.policies.decode import registry

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
    policy = registry.resolve(cfg.decode_policy).from_config(cfg, None)
    assert policy.threshold == 0.85
    assert len(policy.schedule) == 32
