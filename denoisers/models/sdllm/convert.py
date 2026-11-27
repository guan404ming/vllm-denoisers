import argparse
import base64
import json
import math
import pickle
import types
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from safetensors.torch import save_file
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast
from transformers.convert_slow_tokenizer import TikTokenConverter

from denoisers.models.sdllm.config import SDLLMConfig

PATTERN = (
    r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+"
    r"|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*"
    r"|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<mask>"]
NUM_SPECIAL_TOKENS = 1000
VOCAB_SIZE = 131072
ALLOWED_GLOBALS = {
    ("builtins", "dict"),
    ("builtins", "list"),
    ("builtins", "int"),
    ("__builtin__", "dict"),
    ("__builtin__", "list"),
    ("__builtin__", "long"),
    ("collections", "OrderedDict"),
    ("collections", "defaultdict"),
    ("_codecs", "encode"),
    ("torch._utils", "_rebuild_tensor_v2"),
}
IGNORED_MODULES = ("dataloader", "omegaconf", "tiktoken", "typing")


class _Ignored:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in ALLOWED_GLOBALS:
            return super().find_class(module, name)
        if module.split(".")[0] in IGNORED_MODULES:
            return _Ignored
        raise pickle.UnpicklingError(f"{module}.{name} is not allowed in an SDLLM checkpoint.")


_restricted = types.ModuleType("restricted")
_restricted.Unpickler = _Unpickler


def load_state(path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(
        path, map_location="cpu", weights_only=False, pickle_module=_restricted, mmap=True
    )
    prefix = "backbone."
    return {
        k[len(prefix) :]: v for k, v in checkpoint["state_dict"].items() if k.startswith(prefix)
    }


def _condition(state) -> torch.Tensor:
    half = state["sigma_map.mlp.0.weight"].shape[1] // 2
    frequencies = torch.cat([torch.ones(half), torch.zeros(half)]).double()
    hidden = F.silu(
        F.linear(
            frequencies,
            state["sigma_map.mlp.0.weight"].double(),
            state["sigma_map.mlp.0.bias"].double(),
        )
    )
    return F.silu(
        F.linear(
            hidden, state["sigma_map.mlp.2.weight"].double(), state["sigma_map.mlp.2.bias"].double()
        )
    )


def _modulation(state, name, condition, chunks) -> tuple[torch.Tensor, ...]:
    out = F.linear(condition, state[f"{name}.weight"].double(), state[f"{name}.bias"].double())
    return out.chunk(chunks)


def fold(state: dict[str, torch.Tensor], num_layers: int) -> dict[str, torch.Tensor]:
    condition = _condition(state)
    out = {
        "model.embed_tokens.weight": state["vocab_embed.embedding"],
        "lm_head.weight": state["output_layer.linear.weight"],
        "lm_head.bias": state["output_layer.linear.bias"],
    }
    for index in range(num_layers):
        block, layer = f"blocks.{index}", f"model.layers.{index}"
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = _modulation(
            state, f"{block}.adaLN_modulation", condition, 6
        )
        out[f"{layer}.input_layernorm.weight"] = state[f"{block}.norm1.weight"].double() * (
            1 + scale_attn
        )
        out[f"{layer}.input_layernorm.bias"] = shift_attn
        out[f"{layer}.self_attn.qkv_proj.weight"] = state[f"{block}.attn_qkv.weight"]
        out[f"{layer}.self_attn.o_proj.weight"] = (
            gate_attn[:, None] * state[f"{block}.attn_out.weight"].double()
        )
        out[f"{layer}.post_attention_layernorm.weight"] = state[
            f"{block}.norm2.weight"
        ].double() * (1 + scale_mlp)
        out[f"{layer}.post_attention_layernorm.bias"] = shift_mlp
        out[f"{layer}.mlp.fc1.weight"] = state[f"{block}.mlp.0.weight"]
        out[f"{layer}.mlp.fc1.bias"] = state[f"{block}.mlp.0.bias"]
        out[f"{layer}.mlp.fc2.weight"] = gate_mlp[:, None] * state[f"{block}.mlp.2.weight"].double()
        out[f"{layer}.mlp.fc2.bias"] = gate_mlp * state[f"{block}.mlp.2.bias"].double()
    shift, scale = _modulation(state, "output_layer.adaLN_modulation", condition, 2)
    out["model.norm.weight"] = state["output_layer.norm_final.weight"].double() * (1 + scale)
    out["model.norm.bias"] = shift
    return out


class _RankConverter(TikTokenConverter):
    def load_tiktoken_bpe(self, ranks):
        return ranks


def build_tokenizer(ranks: dict[bytes, int]) -> PreTrainedTokenizerFast:
    shifted = {token: rank + NUM_SPECIAL_TOKENS for token, rank in ranks.items()}
    backend = json.loads(_RankConverter(vocab_file=shifted, pattern=PATTERN).converted().to_str())
    specials = SPECIAL_TOKENS + [
        f"<SPECIAL_{index}>" for index in range(len(SPECIAL_TOKENS), NUM_SPECIAL_TOKENS)
    ]
    backend["model"]["vocab"] |= {token: index for index, token in enumerate(specials)}
    backend["added_tokens"] = [
        {
            "id": index,
            "content": token,
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
        for index, token in enumerate(specials)
    ]
    return PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_str(json.dumps(backend)),
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        mask_token="<mask>",
    )


def load_ranks(path, limit: int) -> dict[bytes, int]:
    entries = json.loads(Path(path).read_text())[:limit]
    return {base64.b64decode(entry["token_bytes"]): entry["rank"] for entry in entries}


def convert(source, output) -> None:
    source, output = Path(source), Path(output)
    release = yaml.safe_load((source / "config.yaml").read_text())
    algo, model = release["algo"], release["model"]
    if (
        algo["name"] != "mdlm"
        or algo["time_conditioning"]
        or algo["causal_attention"]
        or not algo["adaLN"]
    ):
        raise ValueError("Only the MDLM release without time conditioning is supported.")
    state = load_state(source / "model.ckpt")
    config = SDLLMConfig(
        vocab_size=state["vocab_embed.embedding"].shape[0],
        hidden_size=model["hidden_size"],
        intermediate_size=state["blocks.0.mlp.0.weight"].shape[0],
        num_hidden_layers=model["n_blocks"],
        num_attention_heads=model["n_heads"],
        max_position_embeddings=model["length"],
        architectures=["SDLLMForMaskedDiffusion"],
        bos_token_id=1,
        eos_token_id=2,
        torch_dtype="bfloat16",
    )
    if config.vocab_size != VOCAB_SIZE or not math.isclose(
        state["rotary_emb.inv_freq"][1].item(),
        config.rope_theta ** (-2 / (config.hidden_size // config.num_attention_heads)),
        rel_tol=1e-6,
    ):
        raise ValueError("The checkpoint vocabulary or rotary base does not match this release.")
    weights = fold(state, config.num_hidden_layers)
    output.mkdir(parents=True, exist_ok=True)
    save_file(
        {
            name: tensor.to(torch.float32 if "norm" in name else torch.bfloat16).contiguous()
            for name, tensor in weights.items()
        },
        output / "model.safetensors",
        metadata={"format": "pt"},
    )
    config.to_json_file(output / "config.json")
    ranks = load_ranks(source / "tokenizer" / "tiktokenizer.json", VOCAB_SIZE - NUM_SPECIAL_TOKENS)
    build_tokenizer(ranks).save_pretrained(output)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert an SDLLM MDLM release to a Hugging Face folder."
    )
    parser.add_argument("source", help="Folder with config.yaml, model.ckpt, and tokenizer/.")
    parser.add_argument(
        "output", help="Folder for config.json, model.safetensors, and the tokenizer."
    )
    args = parser.parse_args()
    convert(args.source, args.output)


if __name__ == "__main__":
    main()
