import math

import torch
import torch.nn.functional as F
from torch import nn
from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding
from vllm.model_executor.models.llama import LlamaAttention, LlamaForCausalLM
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper

from denoisers.models.common.mixin import BlockDiffusionModel


class SigmaRotaryEmbedding(RotaryEmbedding):
    def __init__(self, config):
        rope = config.rope_parameters
        self.original_max = float(rope["original_max_position_embeddings"])
        self.beta_fast = float(rope["beta_fast"])
        self.beta_slow = float(rope["beta_slow"])
        super().__init__(
            config.head_dim,
            config.head_dim,
            config.max_position_embeddings,
            float(rope["rope_theta"]),
            True,
            torch.get_default_dtype(),
        )

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        dim = self.rotary_dim
        factor = self.max_position_embeddings / self.original_max
        pos_freqs = base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)

        def correction(rotations):
            return (
                dim * math.log(self.original_max / (rotations * 2 * math.pi)) / (2 * math.log(base))
            )

        low = max(math.floor(correction(self.beta_fast)), 0)
        high = min(math.ceil(correction(self.beta_slow)), dim - 1)
        ramp = (torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 0.001)
        keep = 1.0 - ramp.clamp(0.0, 1.0)
        return (1.0 - keep) / (factor * pos_freqs) + keep / pos_freqs


def use_sigma_rope(model) -> None:
    rope = SigmaRotaryEmbedding(model.config)
    for module in model.model.modules():
        if isinstance(module, LlamaAttention):
            module.rotary_emb = rope


class NemotronLabsSigmaBlockDiffusion(BlockDiffusionModel, LlamaForCausalLM):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"encoder.": "model.", "diffusion_head.": "lm_head."}
    )

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        config = self.config
        use_sigma_rope(self)
        dim = config.embed_dim
        self.replaid_embedding_matrix = nn.ParameterDict(
            {"matrix": nn.Parameter(torch.empty(config.vocab_size, dim), requires_grad=False)}
        )
        self.gamma_bounds = nn.ParameterDict(
            {"endpoint_values": nn.Parameter(torch.empty(2), requires_grad=False)}
        )
        self.replaid_input_linear = nn.Linear(dim, config.hidden_size, bias=False)
        self.replaid_selfcond_linear = nn.Linear(dim, config.hidden_size, bias=False)
        self.replaid_gamma_linear = nn.Linear(64, config.hidden_size, bias=False)

    def gamma_grid(self, calls: int) -> torch.Tensor:
        low, high = self.gamma_bounds["endpoint_values"].double()
        return torch.linspace(high, low, calls, dtype=torch.float64, device=low.device)

    def embedding_matrix(self) -> torch.Tensor:
        matrix = self.replaid_embedding_matrix["matrix"].float()
        return matrix / (torch.linalg.norm(matrix, dim=1, keepdim=True) + 1e-8)

    def latent_input(self, z: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        dim = self.config.embed_dim
        scale = (torch.sigmoid(-gamma) / dim + torch.sigmoid(gamma)).sqrt()
        hidden = F.linear(z / scale, self.replaid_input_linear.weight.float())
        angles = torch.linspace(-5.0, 5.0, 32, device=z.device).exp() * gamma
        time = torch.cat((angles.sin(), angles.cos()), -1)
        return hidden + F.linear(time, self.replaid_gamma_linear.weight.float())

    def prior(self, z: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        scaled = torch.sigmoid(-gamma).sqrt() / torch.sigmoid(gamma) * z
        return scaled @ self.embedding_matrix().t()

    def self_conditioning(self, embeds: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:
        dim = self.config.embed_dim
        step = soft[:, -1].float().round().long()
        grid = self.gamma_grid(self.config.denoise_steps + 1)
        gamma = grid[(step - 1).clamp(min=0)].float().unsqueeze(1)
        hidden = self.latent_input(soft[:, :dim].float(), gamma)
        selfcond = soft[:, dim : 2 * dim].float() * math.sqrt(dim)
        hidden = hidden + F.linear(selfcond, self.replaid_selfcond_linear.weight.float())
        return torch.where(step.unsqueeze(1) > 0, hidden.to(embeds.dtype), embeds)

    def unconditional(self, z: torch.Tensor, gamma: torch.Tensor) -> torch.Tensor:
        rows, width, _ = z.shape
        hidden = self.latent_input(z, gamma).to(self.lm_head.weight.dtype).flatten(0, 1)
        positions = torch.arange(width, device=z.device).repeat(rows)
        residual = None
        for layer in self.model.layers:
            if residual is None:
                residual = hidden
                hidden = layer.input_layernorm(hidden)
            else:
                hidden, residual = layer.input_layernorm(hidden, residual)
            attn = layer.self_attn
            qkv = attn.qkv_proj(hidden)[0]
            q, k, v = qkv.split([attn.q_size, attn.kv_size, attn.kv_size], -1)
            q, k = attn.rotary_emb(positions, q, k)
            q, k, v = (t.view(rows, width, -1, attn.head_dim).transpose(1, 2) for t in (q, k, v))
            out = F.scaled_dot_product_attention(q, k, v, scale=attn.scaling, enable_gqa=True)
            hidden = attn.o_proj(out.transpose(1, 2).reshape(rows * width, -1))[0]
            hidden, residual = layer.post_attention_layernorm(hidden, residual)
            hidden = layer.mlp(hidden)
        hidden, _ = self.model.norm(hidden, residual)
        logits = self.compute_logits(hidden).float().view(rows, width, -1)
        return logits + self.prior(z, gamma)

    def load_weights(self, weights):
        loader = AutoWeightsLoader(self, ignore_unexpected_prefixes=["ar_head.", "noise_schedule."])
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
