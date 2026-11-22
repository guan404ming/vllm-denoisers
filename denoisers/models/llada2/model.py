import torch
from vllm.model_executor.layers.fused_moe.router.custom_routing_router import (
    CustomRoutingRouter,
)
from vllm.model_executor.models.bailing_moe import BailingMoE, BailingMoeV2ForCausalLM
from vllm.model_executor.models.config import VerifyAndUpdateConfig

from denoisers.models.llada2.config import LLaDA2MoeConfig
from denoisers.models.llada2.routing import BlockRouter
from denoisers.models.mixin import BlockDiffusionModel
from denoisers.models.verify import verify_block_diffusion


class LLaDA2MoeModelLMConfig(VerifyAndUpdateConfig):
    @classmethod
    def verify_and_update_config(cls, vllm_config) -> None:
        hf_config = vllm_config.model_config.hf_config
        if not isinstance(hf_config, LLaDA2MoeConfig):
            raise TypeError("LLaDA2 needs denoisers's config; drop --trust-remote-code.")
        if hf_config.block_size not in (None, hf_config.canvas_length):
            raise ValueError(f"LLaDA2 block_size must equal canvas_length {hf_config.canvas_length}.")
        if hf_config.decode_policy is None:
            hf_config.decode_policy = "indel" if hf_config.expert_capacity else "low_confidence"
        if hf_config.expert_capacity and vllm_config.parallel_config.enable_eplb:
            raise ValueError("LLaDA2.2 block routing does not support EPLB.")
        if hf_config.expert_capacity and vllm_config.model_config.quantization is not None:
            raise ValueError("LLaDA2.2 block routing needs unquantized MoE kernels.")
        verify_block_diffusion(vllm_config, hf_config, "LLaDA2")


class FP32RouterMoE(BailingMoE):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_size = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_size)
        router_logits, _ = self.gate(hidden_states)
        out = self.experts(hidden_states=hidden_states, router_logits=router_logits)
        return out.view(num_tokens, hidden_size)


class LLaDA2MoeModelLM(BlockDiffusionModel, BailingMoeV2ForCausalLM):
    def __init__(self, *, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.block_routers: list[BlockRouter] = []
        config = self.config
        for layer in self.model.layers:
            moe = getattr(layer, "mlp", None)
            if not isinstance(moe, BailingMoE):
                continue
            moe.__class__ = FP32RouterMoE
            if config.expert_capacity:
                router = BlockRouter(
                    moe.gate,
                    config.canvas_length,
                    config.expert_capacity,
                    config.routed_scaling_factor,
                )
                moe.experts.router = CustomRoutingRouter(
                    top_k=moe.top_k,
                    global_num_experts=moe.num_experts,
                    custom_routing_function=router,
                    eplb_state=moe.experts.router.eplb_state,
                    renormalize=False,
                )
                self.block_routers.append(router)
