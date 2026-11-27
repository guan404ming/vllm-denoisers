from vllm.model_executor.models.config import VerifyAndUpdateConfig

from denoisers.models.common.verify import verify_block_diffusion


class SigmaModelConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config):
        config = model_config.hf_config
        decoding = model_config.try_get_generation_config().get("decoding")
        config.decode_policy = None if decoding == "self_speculative" else "sigma"
        if config.decode_policy is None:
            model_config.override_generation_config = {
                "temperature": 0.0,
                **model_config.override_generation_config,
            }
            return
        distilled = getattr(config, "native_nfe", None) is not None
        config.canvas_length = config.block_size
        config.feedback_policy = "latent_conditioning"
        config.mask_token_id = -1
        config.denoise_steps = config.native_nfe if distilled else 32
        config.guidance_scale = 1.0 if distilled else 3.0
        config.score_temperature = 1.0 if distilled else 0.3

    @staticmethod
    def verify_and_update_config(vllm_config):
        config = vllm_config.model_config.hf_config
        if config.decode_policy is not None:
            verify_block_diffusion(vllm_config, config, "Sigma")
