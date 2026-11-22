from transformers import PretrainedConfig

from denoisers.models.config import DecodeConfig


class LLaDA2MoeConfig(DecodeConfig, PretrainedConfig):
    model_type = "llada2_moe"

    def __init__(
        self,
        use_qk_norm: bool = True,
        use_sliding_window: bool = False,
        sliding_window: int | None = None,
        canvas_length: int = 32,
        block_size: int | None = None,
        mask_token_id: int = 156895,
        delete_token_id: int = 156930,
        split_token_id: int = 156931,
        expert_capacity: int | None = None,
        decode_policy: str | None = None,
        temperature: float = 0.0,
        denoise_steps: int = 32,
        confidence_threshold: float | None = None,
        editing_threshold: float | None = None,
        max_post_steps: int = 16,
        max_steps_per_block: int = 1000,
        **kwargs,
    ):
        self.use_qk_norm = use_qk_norm
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if use_sliding_window else None
        self.canvas_length = canvas_length
        self.block_size = block_size
        self.mask_token_id = mask_token_id
        self.delete_token_id = delete_token_id
        self.split_token_id = split_token_id
        self.expert_capacity = expert_capacity
        self.decode_policy = decode_policy
        self.temperature = temperature
        self.denoise_steps = denoise_steps
        self.confidence_threshold = confidence_threshold
        self.editing_threshold = editing_threshold
        self.max_post_steps = max_post_steps
        self.max_steps_per_block = max_steps_per_block
        super().__init__(**kwargs)

    def verify(self, vllm_config) -> None:
        if self.block_size not in (None, self.canvas_length):
            raise ValueError(f"LLaDA2 block_size must equal canvas_length {self.canvas_length}.")
        if self.decode_policy is None:
            self.decode_policy = "indel" if self.expert_capacity else "low_confidence"
        if self.expert_capacity and vllm_config.parallel_config.enable_eplb:
            raise ValueError("LLaDA2.2 block routing does not support EPLB.")
        if self.expert_capacity and vllm_config.model_config.quantization is not None:
            raise ValueError("LLaDA2.2 block routing needs unquantized MoE kernels.")
