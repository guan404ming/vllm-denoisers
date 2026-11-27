from transformers import Qwen2Config, Qwen3Config

from denoisers.models.sdar.config import SDARDecodeConfig

MDLM = {"decode_policy": "llada", "canvas_length": 256, "sub_block_length": 64}


class A2DDecodeConfig(SDARDecodeConfig):
    def __init__(self, canvas_length: int = 32, confidence_threshold: float = 2.0, **kwargs):
        super().__init__(
            canvas_length=canvas_length, confidence_threshold=confidence_threshold, **kwargs
        )

    def verify(self, vllm_config) -> None:
        model_config = vllm_config.model_config
        overrides = model_config.hf_overrides
        overrides = overrides if isinstance(overrides, dict) else {}
        if "mdlm" in model_config.model.lower():
            for key, value in MDLM.items():
                if key not in overrides:
                    setattr(self, key, value)
        super().verify(vllm_config)


class A2DQwen3Config(A2DDecodeConfig, Qwen3Config):
    model_type = "a2d-qwen3"
    __init__ = A2DDecodeConfig.__init__


class A2DQwen2Config(A2DDecodeConfig, Qwen2Config):
    model_type = "a2d-qwen2"

    def __init__(self, mask_token_id: int = 151665, **kwargs):
        super().__init__(mask_token_id=mask_token_id, **kwargs)
