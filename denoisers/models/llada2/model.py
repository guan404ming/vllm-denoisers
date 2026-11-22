import torch
from vllm.model_executor.layers.fused_moe.router.custom_routing_router import (
    CustomRoutingRouter,
)
from vllm.model_executor.models.bailing_moe import BailingMoE, BailingMoeV2ForCausalLM

from denoisers.models.llada2.routing import BlockRouter
from denoisers.models.mixin import BlockDiffusionModel


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
