from transformers import LlamaConfig

from denoisers.models.config import DecodeConfig


class NemotronLabsDiffusionConfig(DecodeConfig, LlamaConfig):
    model_type = "nemotron_labs_diffusion"

    def __init__(
        self,
        block_size: int = 32,
        canvas_length: int | None = None,
        mask_token_id: int = 100,
        decode_policy: str = "nemotron_diffusion",
        confidence_threshold: float | None = 0.9,
        temperature: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.block_size = block_size
        self.canvas_length = block_size if canvas_length is None else canvas_length
        self.mask_token_id = mask_token_id
        self.decode_policy = decode_policy
        self.confidence_threshold = confidence_threshold
        self.temperature = temperature
        rope = self.rope_parameters
        if rope.get("llama_4_scaling_beta") is not None:
            self.llama_4_scaling = {
                "beta": rope["llama_4_scaling_beta"],
                "original_max_position_embeddings": rope["original_max_position_embeddings"],
            }

    def verify(self, vllm_config):
        if self.temperature != 0:
            raise ValueError("Nemotron diffusion currently supports greedy sampling.")
