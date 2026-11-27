import json
from pathlib import Path

from denoisers.models.llada.config import LLaDAConfig, LLaDAMoEConfig
from denoisers.policies.decode import registry

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
    assert type(cfg) is LLaDAConfig
    assert cfg.hidden_size == 4096 and cfg.num_hidden_layers == 32
    assert cfg.intermediate_size == 12288 and cfg.num_key_value_heads == 32
    assert cfg.max_position_embeddings == 4096 and cfg.hidden_act == "silu"
    assert cfg.tie_word_embeddings is False
    assert cfg.canvas_length == 256 and cfg.mask_token_id == 126336
    assert cfg.decode_policy == "llada" and cfg.sub_block_length == 32
    policy = registry.resolve(cfg.decode_policy).from_config(cfg, None)
    assert policy.steps_per_block == 32 and policy.last == 255
    assert policy.schedule.tolist() == [1] * 32


MOE_CONFIG = {
    "architectures": ["LLaDAMoEModel"],
    "model_type": "llada",
    "hidden_size": 2048,
    "num_hidden_layers": 16,
    "num_attention_heads": 16,
    "num_key_value_heads": 16,
    "expert_intermediate_size": 1024,
    "num_experts": 64,
    "num_experts_per_tok": 8,
    "norm_topk_prob": None,
    "shared_expert_intermediate_size": None,
    "qk_layernorm": True,
    "rope_theta": 50000,
    "vocab_size": 157184,
    "tie_word_embeddings": False,
    "eos_token_id": 156892,
}


def test_moe_checkpoint_selects_qwen3_moe_fields(tmp_path: Path):
    (tmp_path / "config.json").write_text(json.dumps(MOE_CONFIG))
    cfg = LLaDAConfig.from_pretrained(tmp_path)
    assert isinstance(cfg, LLaDAMoEConfig)
    assert cfg.hidden_size == 2048 and cfg.vocab_size == 157184
    assert cfg.moe_intermediate_size == 1024 and cfg.num_experts == 64
    assert cfg.num_experts_per_tok == 8 and cfg.norm_topk_prob is False
    assert cfg.mlp_only_layers == [] and cfg.decoder_sparse_step == 1
    assert cfg.shared_expert_intermediate_size == 0
    assert cfg.canvas_length == 256 and cfg.mask_token_id == 156895
    assert cfg.decode_policy == "llada" and cfg.sub_block_length == 32
