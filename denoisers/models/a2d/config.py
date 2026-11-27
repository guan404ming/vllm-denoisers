from transformers import Qwen3Config

from denoisers.models.sdar.config import SDARDecodeConfig


class A2DQwen3Config(SDARDecodeConfig, Qwen3Config):
    model_type = "a2d-qwen3"

    def __init__(self, canvas_length: int = 32, confidence_threshold: float = 2.0, **kwargs):
        super().__init__(
            canvas_length=canvas_length, confidence_threshold=confidence_threshold, **kwargs
        )
