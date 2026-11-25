from vllm.model_executor.models.config import Gemma4Config, VerifyAndUpdateConfig
from vllm.model_executor.models.diffusion_gemma import (
    DiffusionGemmaForConditionalGeneration,
)

from denoisers.models.common.mixin import BlockDiffusionModel
from denoisers.models.common.verify import verify_block_diffusion
from denoisers.ops.sampling import given

GENERATION_FIELDS = ("t_min", "t_max", "confidence_threshold", "stability_threshold")


class DiffusionGemmaForBlockDiffusionConfig(VerifyAndUpdateConfig):
    @classmethod
    def verify_and_update_config(cls, vllm_config) -> None:
        Gemma4Config.verify_and_update_config(vllm_config)
        model_config = vllm_config.model_config
        hf_config = model_config.hf_config
        generation = model_config.try_get_generation_config()
        sampler = generation.get("sampler_config") or {}
        if "EntropyBound" not in sampler.get("_cls_name", ""):
            raise ValueError("DiffusionGemma needs an EntropyBound sampler_config.")
        diffusion = vllm_config.diffusion_config
        if diffusion is not None:
            hf_config.canvas_length = diffusion.canvas_length
        for name in GENERATION_FIELDS:
            setattr(hf_config, name, given(getattr(hf_config, name, None), generation[name]))
        hf_config.entropy_bound = given(
            getattr(hf_config, "entropy_bound", None), sampler["entropy_bound"]
        )
        steps = given(
            getattr(hf_config, "max_denoising_steps", None), generation["max_denoising_steps"]
        )
        hf_config.max_denoising_steps = given(
            getattr(diffusion, "max_denoising_steps", None), steps
        )
        hf_config.sparse_self_conditioning = getattr(hf_config, "sparse_self_conditioning", True)
        hf_config.decode_policy = "entropy_bound"
        if not hasattr(hf_config, "feedback_policy"):
            hf_config.feedback_policy = "self_conditioning"
        hf_config.mask_token_id = -1
        if "max_new_tokens" not in model_config.override_generation_config:
            model_config.override_generation_config["max_new_tokens"] = None
        verify_block_diffusion(vllm_config, hf_config, "DiffusionGemma")


class DiffusionGemma(BlockDiffusionModel, DiffusionGemmaForConditionalGeneration):
    def compute_logits(self, hidden_states):
        return self.logits_processor(self.lm_head, hidden_states)
