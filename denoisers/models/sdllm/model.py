import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.model_executor.layers.activation import get_act_fn
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.layernorm import LayerNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.utils import AutoWeightsLoader, maybe_prefix

from denoisers.models.common.mixin import BlockDiffusionModel


def _norm(config) -> LayerNorm:
    return LayerNorm(config.hidden_size, eps=config.layer_norm_eps, dtype=torch.float32)


class SDLLMAttention(nn.Module):
    def __init__(self, config, cache_config, quant_config, prefix: str):
        super().__init__()
        heads = config.num_attention_heads
        head_size = config.hidden_size // heads
        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            head_size,
            heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            config.hidden_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            head_size,
            max_position=config.max_position_embeddings,
            rope_parameters={"rope_theta": config.rope_theta},
        )
        self.attn = Attention(
            heads,
            head_size,
            head_size**-0.5,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        output, _ = self.o_proj(self.attn(q, k, v))
        return output


class SDLLMMLP(nn.Module):
    def __init__(self, config, quant_config, prefix: str):
        super().__init__()
        self.fc1 = ColumnParallelLinear(
            config.hidden_size,
            config.intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.fc1",
        )
        self.act = get_act_fn("gelu_pytorch_tanh")
        self.fc2 = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.fc2",
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, _ = self.fc1(hidden_states)
        hidden_states, _ = self.fc2(self.act(hidden_states))
        return hidden_states


class SDLLMLayer(nn.Module):
    def __init__(self, config, cache_config, quant_config, prefix: str):
        super().__init__()
        self.input_layernorm = _norm(config)
        self.self_attn = SDLLMAttention(config, cache_config, quant_config, f"{prefix}.self_attn")
        self.post_attention_layernorm = _norm(config)
        self.mlp = SDLLMMLP(config, quant_config, f"{prefix}.mlp")

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.self_attn(
            positions, self.input_layernorm(hidden_states)
        )
        return hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))


@support_torch_compile
class SDLLMModel(nn.Module):
    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            SDLLMLayer(
                config,
                vllm_config.cache_config,
                vllm_config.quant_config,
                f"{prefix}.layers.{index}",
            )
            for index in range(config.num_hidden_layers)
        )
        self.norm = _norm(config)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        for layer in self.layers:
            hidden_states = layer(positions, hidden_states)
        return self.norm(hidden_states)


class SDLLMForMaskedLM(nn.Module):
    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__()
        self.config = vllm_config.model_config.hf_config
        self.model = SDLLMModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))
        self.lm_head = ParallelLMHead(
            self.config.vocab_size,
            self.config.hidden_size,
            bias=True,
            quant_config=vllm_config.quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(self.config.vocab_size)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.logits_processor(self.lm_head, hidden_states, self.lm_head.bias)

    def load_weights(self, weights):
        return AutoWeightsLoader(self).load_weights(weights)


class SDLLMForMaskedDiffusion(BlockDiffusionModel, SDLLMForMaskedLM):
    pass
