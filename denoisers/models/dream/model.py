from vllm.model_executor.models.config import VerifyAndUpdateConfig
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM

from denoisers.models.dream.config import DreamConfig
from denoisers.models.mixin import BlockDiffusionModel
from denoisers.models.verify import verify_block_diffusion


class DreamModelConfig(VerifyAndUpdateConfig):
    @classmethod
    def verify_and_update_config(cls, vllm_config) -> None:
        hf_config = vllm_config.model_config.hf_config
        if not isinstance(hf_config, DreamConfig):
            raise TypeError("Dream needs denoisers's config; drop --trust-remote-code.")
        if hf_config.denoise_steps is None:
            hf_config.denoise_steps = hf_config.canvas_length
        verify_block_diffusion(vllm_config, hf_config, "Dream")


class DreamModel(BlockDiffusionModel, Qwen2ForCausalLM):
    pass
