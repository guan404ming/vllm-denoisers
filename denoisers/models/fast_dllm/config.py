from transformers import Qwen2Config

from denoisers.models.config import DecodeConfig


class FastDLLMQwenConfig(DecodeConfig, Qwen2Config):
    model_type = "Fast_dLLM_Qwen"

    def __init__(
        self,
        bd_size: int = 32,
        canvas_length: int = 32,
        mask_token_id: int = 151665,
        decode_policy: str = "fast_dllm",
        temperature: float = 0.0,
        small_block_size: int = 8,
        confidence_threshold: float | None = None,
        **kwargs,
    ):
        self.bd_size = bd_size
        self.canvas_length = canvas_length
        self.mask_token_id = mask_token_id
        self.decode_policy = decode_policy
        self.temperature = temperature
        self.small_block_size = small_block_size
        self.confidence_threshold = confidence_threshold
        super().__init__(**kwargs)

    def verify(self, vllm_config) -> None:
        if self.canvas_length != self.bd_size:
            raise ValueError(f"Fast-dLLM v2 canvas_length must equal bd_size {self.bd_size}.")
        if self.canvas_length % self.small_block_size:
            raise ValueError("Fast-dLLM v2 small_block_size must divide canvas_length.")
