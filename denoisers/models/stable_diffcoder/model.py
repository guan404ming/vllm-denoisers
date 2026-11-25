from vllm.model_executor.models.config import VerifyAndUpdateConfig
from vllm.model_executor.models.llama import LlamaForCausalLM

from denoisers.models.common.mixin import BlockDiffusionModel
from denoisers.models.common.verify import verify_block_diffusion
from denoisers.ops.sampling import given


class StableDiffcoderModelConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config):
        config = model_config.hf_config
        config.canvas_length = getattr(config, "canvas_length", 4)
        config.mask_token_id = given(getattr(config, "mask_token_id", None), 5)
        config.decode_policy = getattr(config, "decode_policy", "low_confidence")
        config.denoise_steps = given(getattr(config, "denoise_steps", None), config.canvas_length)
        config.temperature = getattr(config, "temperature", 0.0)
        config.confidence_threshold = given(getattr(config, "confidence_threshold", None), 1.0)

    @staticmethod
    def verify_and_update_config(vllm_config):
        verify_block_diffusion(vllm_config, vllm_config.model_config.hf_config, "Stable-DiffCoder")


class StableDiffcoderForCausalLM(BlockDiffusionModel, LlamaForCausalLM):
    pass
