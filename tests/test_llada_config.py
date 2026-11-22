import json
from pathlib import Path

from denoisers.models.llada.config import LLaDAConfig
from denoisers.policies import POLICIES

CONFIG = {
    "architectures": ["LLaDAModelLM"],
    "model_type": "llada",
    "d_model": 4096,
    "n_layers": 32,
    "n_heads": 32,
    "n_kv_heads": 32,
    "mlp_hidden_size": 12288,
    "max_sequence_length": 4096,
    "vocab_size": 126464,
    "rms_norm_eps": 1e-5,
    "rope_theta": 500000.0,
    "activation_type": "silu",
    "weight_tying": False,
    "mask_token_id": 126336,
    "eos_token_id": 126081,
}


def test_checkpoint_config_maps_llama_fields(tmp_path: Path):
    (tmp_path / "config.json").write_text(json.dumps(CONFIG))
    cfg = LLaDAConfig.from_pretrained(tmp_path)
    assert cfg.hidden_size == 4096 and cfg.num_hidden_layers == 32
    assert cfg.intermediate_size == 12288 and cfg.num_key_value_heads == 32
    assert cfg.max_position_embeddings == 4096 and cfg.hidden_act == "silu"
    assert cfg.tie_word_embeddings is False
    assert cfg.canvas_length == 256 and cfg.mask_token_id == 126336
    assert cfg.decode_policy == "llada" and cfg.sub_block_length == 32
    policy = POLICIES[cfg.decode_policy].from_config(cfg, None)
    assert policy.steps_per_block == 32 and policy.last == 255
    assert policy.schedule.tolist() == [1] * 32
