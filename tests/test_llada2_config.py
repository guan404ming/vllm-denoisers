import json
from pathlib import Path

from denoisers.models.llada2.config import LLaDA2MoeConfig

CONFIG = {
    "architectures": ["LLaDA2MoeModelLM"],
    "model_type": "llada2_moe",
    "hidden_size": 2048,
    "num_experts": 256,
    "num_hidden_layers": 20,
    "sliding_window": 4096,
    "use_sliding_window": False,
    "router_dtype": "fp32",
    "tie_word_embeddings": False,
    "vocab_size": 157184,
}


def test_checkpoint_config_fixes(tmp_path: Path):
    (tmp_path / "config.json").write_text(json.dumps(CONFIG))
    cfg = LLaDA2MoeConfig.from_pretrained(tmp_path)
    assert cfg.sliding_window is None
    assert cfg.use_qk_norm is True
    assert cfg.canvas_length == 32
    assert cfg.mask_token_id == 156895
    assert cfg.decode_policy is None and cfg.confidence_threshold is None
    assert cfg.expert_capacity is None
    assert cfg.num_experts == 256
    assert cfg.router_dtype == "fp32"
    assert cfg.tie_word_embeddings is False


def test_sliding_window_kept_when_enabled():
    cfg = LLaDA2MoeConfig(use_sliding_window=True, sliding_window=4096)
    assert cfg.sliding_window == 4096
