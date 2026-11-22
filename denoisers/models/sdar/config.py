from transformers import Qwen3Config, Qwen3MoeConfig

from denoisers.models.config import DecodeConfig


class SDARDecodeConfig(DecodeConfig):
    def __init__(
        self,
        canvas_length: int = 4,
        mask_token_id: int = 151669,
        decode_policy: str = "low_confidence",
        temperature: float = 0.0,
        denoise_steps: int | None = None,
        confidence_threshold: float = 0.85,
        **kwargs,
    ):
        self.canvas_length = canvas_length
        self.mask_token_id = mask_token_id
        self.decode_policy = decode_policy
        self.temperature = temperature
        self.denoise_steps = denoise_steps
        self.confidence_threshold = confidence_threshold
        super().__init__(**kwargs)


class SDARConfig(SDARDecodeConfig, Qwen3Config):
    model_type = "sdar"
    __init__ = SDARDecodeConfig.__init__


class SDARMoeConfig(SDARDecodeConfig, Qwen3MoeConfig):
    model_type = "sdar_moe"
    __init__ = SDARDecodeConfig.__init__
