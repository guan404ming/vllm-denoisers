from transformers import Qwen2Config

from denoisers.models.common.config import DecodeConfig


class DreamConfig(DecodeConfig, Qwen2Config):
    model_type = "Dream"

    def __init__(
        self,
        canvas_length: int = 256,
        mask_token_id: int = 151666,
        decode_policy: str = "dream",
        temperature: float = 0.0,
        top_p: float = 1.0,
        denoise_steps: int | None = None,
        eps: float = 1e-3,
        **kwargs,
    ):
        self.canvas_length = canvas_length
        self.mask_token_id = mask_token_id
        self.decode_policy = decode_policy
        self.temperature = temperature
        self.top_p = top_p
        self.denoise_steps = denoise_steps
        self.eps = eps
        super().__init__(**kwargs)
