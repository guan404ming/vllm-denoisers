import os
import pickle
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from denoisers.models.sdllm.config import SDLLMConfig
from denoisers.models.sdllm.convert import build_tokenizer, fold, load_state
from denoisers.policies.decode import registry

DIM, HEADS, COND, INNER, VOCAB, LAYERS = 8, 2, 4, 16, 10, 2


def _release_state() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    shapes = {
        "vocab_embed.embedding": (VOCAB, DIM),
        "sigma_map.mlp.0.weight": (COND, 8),
        "sigma_map.mlp.0.bias": (COND,),
        "sigma_map.mlp.2.weight": (COND, COND),
        "sigma_map.mlp.2.bias": (COND,),
        "output_layer.norm_final.weight": (DIM,),
        "output_layer.linear.weight": (VOCAB, DIM),
        "output_layer.linear.bias": (VOCAB,),
        "output_layer.adaLN_modulation.weight": (2 * DIM, COND),
        "output_layer.adaLN_modulation.bias": (2 * DIM,),
    }
    for index in range(LAYERS):
        block = f"blocks.{index}"
        shapes |= {
            f"{block}.norm1.weight": (DIM,),
            f"{block}.attn_qkv.weight": (3 * DIM, DIM),
            f"{block}.attn_out.weight": (DIM, DIM),
            f"{block}.norm2.weight": (DIM,),
            f"{block}.mlp.0.weight": (INNER, DIM),
            f"{block}.mlp.0.bias": (INNER,),
            f"{block}.mlp.2.weight": (DIM, INNER),
            f"{block}.mlp.2.bias": (DIM,),
            f"{block}.adaLN_modulation.weight": (6 * DIM, COND),
            f"{block}.adaLN_modulation.bias": (6 * DIM,),
        }
    return {name: torch.randn(shape, dtype=torch.float64) for name, shape in shapes.items()}


def _attend(qkv: torch.Tensor) -> torch.Tensor:
    q, k, v = (t.unflatten(-1, (HEADS, -1)).transpose(0, 1) for t in qkv.chunk(3, dim=-1))
    return F.scaled_dot_product_attention(q, k, v).transpose(0, 1).flatten(1)


def _reference(s, tokens):
    time = torch.cat([torch.cos(torch.zeros(4)), torch.sin(torch.zeros(4))]).double()
    hidden = F.silu(F.linear(time, s["sigma_map.mlp.0.weight"], s["sigma_map.mlp.0.bias"]))
    c = F.silu(F.linear(hidden, s["sigma_map.mlp.2.weight"], s["sigma_map.mlp.2.bias"]))
    x = s["vocab_embed.embedding"][tokens]
    for index in range(LAYERS):
        b = f"blocks.{index}"
        mod = F.linear(c, s[f"{b}.adaLN_modulation.weight"], s[f"{b}.adaLN_modulation.bias"])
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = mod.chunk(6)
        h = F.layer_norm(x, (DIM,)) * s[f"{b}.norm1.weight"] * (1 + scale_a) + shift_a
        x = x + gate_a * (_attend(h @ s[f"{b}.attn_qkv.weight"].T) @ s[f"{b}.attn_out.weight"].T)
        h = F.layer_norm(x, (DIM,)) * s[f"{b}.norm2.weight"] * (1 + scale_m) + shift_m
        h = F.gelu(F.linear(h, s[f"{b}.mlp.0.weight"], s[f"{b}.mlp.0.bias"]), approximate="tanh")
        x = x + gate_m * F.linear(h, s[f"{b}.mlp.2.weight"], s[f"{b}.mlp.2.bias"])
    mod = F.linear(
        c, s["output_layer.adaLN_modulation.weight"], s["output_layer.adaLN_modulation.bias"]
    )
    shift, scale = mod.chunk(2)
    x = F.layer_norm(x, (DIM,)) * s["output_layer.norm_final.weight"] * (1 + scale) + shift
    return F.linear(x, s["output_layer.linear.weight"], s["output_layer.linear.bias"])


def _folded(w, tokens):
    x = w["model.embed_tokens.weight"][tokens]
    for index in range(LAYERS):
        p = f"model.layers.{index}"
        h = F.layer_norm(
            x, (DIM,), w[f"{p}.input_layernorm.weight"], w[f"{p}.input_layernorm.bias"]
        )
        x = (
            x
            + _attend(h @ w[f"{p}.self_attn.qkv_proj.weight"].T)
            @ w[f"{p}.self_attn.o_proj.weight"].T
        )
        h = F.layer_norm(
            x,
            (DIM,),
            w[f"{p}.post_attention_layernorm.weight"],
            w[f"{p}.post_attention_layernorm.bias"],
        )
        h = F.gelu(
            F.linear(h, w[f"{p}.mlp.fc1.weight"], w[f"{p}.mlp.fc1.bias"]), approximate="tanh"
        )
        x = x + F.linear(h, w[f"{p}.mlp.fc2.weight"], w[f"{p}.mlp.fc2.bias"])
    x = F.layer_norm(x, (DIM,), w["model.norm.weight"], w["model.norm.bias"])
    return F.linear(x, w["lm_head.weight"], w["lm_head.bias"])


def test_folded_weights_match_the_adaln_model():
    state = _release_state()
    weights = {name: tensor.double() for name, tensor in fold(state, LAYERS).items()}
    tokens = torch.tensor([1, 4, 9, 2, 0])
    torch.testing.assert_close(_folded(weights, tokens), _reference(state, tokens))


def test_checkpoint_loader_keeps_backbone_and_rejects_code(tmp_path: Path):
    tensor = torch.arange(3.0)
    torch.save({"state_dict": {"backbone.a": tensor, "ema.a": tensor}}, tmp_path / "ok.ckpt")
    assert load_state(tmp_path / "ok.ckpt").keys() == {"a"}
    torch.save({"state_dict": {}, "hook": os.system}, tmp_path / "bad.ckpt")
    with pytest.raises(pickle.UnpicklingError):
        load_state(tmp_path / "bad.ckpt")


def test_tokenizer_places_ranks_after_special_tokens():
    ranks = {bytes([index]): index for index in range(256)} | {b"ab": 256}
    tokenizer = build_tokenizer(ranks)
    assert len(tokenizer) == 1000 + 257
    assert tokenizer.encode("ab") == [1256]
    assert tokenizer.mask_token_id == 3 and tokenizer.eos_token_id == 2
    assert tokenizer.decode([1256, 1000 + ord("c")]) == "abc"


def test_config_defaults_resolve_the_llada_policy(tmp_path: Path):
    SDLLMConfig(architectures=["SDLLMForMaskedDiffusion"]).to_json_file(tmp_path / "config.json")
    cfg = SDLLMConfig.from_pretrained(tmp_path)
    assert cfg.model_type == "sdllm" and cfg.tie_word_embeddings is False
    assert cfg.canvas_length == 256 and cfg.mask_token_id == 3
    policy = registry.resolve(cfg.decode_policy).from_config(cfg, None)
    assert policy.steps_per_block == 32 and policy.last == 255
