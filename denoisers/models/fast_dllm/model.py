from vllm.model_executor.models.config import VerifyAndUpdateConfig
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM

from denoisers.models.fast_dllm.config import FastDLLMQwenConfig
from denoisers.models.mixin import BlockDiffusionModel
from denoisers.models.verify import verify_block_diffusion


class FastDLLMQwenForCausalLMConfig(VerifyAndUpdateConfig):
    @classmethod
    def verify_and_update_config(cls, vllm_config) -> None:
        hf_config = vllm_config.model_config.hf_config
        if not isinstance(hf_config, FastDLLMQwenConfig):
            raise TypeError("Fast-dLLM v2 needs denoisers's config; drop --trust-remote-code.")
        if hf_config.canvas_length != hf_config.bd_size:
            raise ValueError(f"Fast-dLLM v2 canvas_length must equal bd_size {hf_config.bd_size}.")
        if hf_config.canvas_length % hf_config.small_block_size:
            raise ValueError("Fast-dLLM v2 small_block_size must divide canvas_length.")
        verify_block_diffusion(vllm_config, hf_config, "Fast-dLLM v2")


class FastDLLMQwenForCausalLM(BlockDiffusionModel, Qwen2ForCausalLM):
    pass
