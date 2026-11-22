from vllm.model_executor.models.config import VerifyAndUpdateConfig
from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM

from denoisers.models.mixin import BlockDiffusionModel
from denoisers.models.sdar.config import SDARDecodeConfig
from denoisers.models.verify import verify_block_diffusion


class SDARForCausalLMConfig(VerifyAndUpdateConfig):
    @classmethod
    def verify_and_update_config(cls, vllm_config) -> None:
        hf_config = vllm_config.model_config.hf_config
        if not isinstance(hf_config, SDARDecodeConfig):
            raise TypeError("SDAR needs denoisers's config; drop --trust-remote-code.")
        if hf_config.denoise_steps is None:
            hf_config.denoise_steps = hf_config.canvas_length
        verify_block_diffusion(vllm_config, hf_config, "SDAR")


class SDARForCausalLM(BlockDiffusionModel, Qwen3ForCausalLM):
    pass


class SDARMoeForCausalLM(BlockDiffusionModel, Qwen3MoeForCausalLM):
    pass
