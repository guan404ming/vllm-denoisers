from transformers import LlamaConfig, Mistral3Config, PreTrainedTokenizerFast

from denoisers.models.common.config import DecodeConfig


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


class NemotronLabsDiffusionVLMConfig(DecodeConfig, Mistral3Config):
    model_type = "nemotron_labs_diffusion_vlm"

    def __init__(
        self, text_config=None, block_size=32, mask_token_id=100, confidence_threshold=0.9, **kwargs
    ):
        if text_config is None:
            text_config = NemotronLabsDiffusionConfig(
                **{**kwargs, "architectures": ["MistralForCausalLM"]}
            )
        for name in ("rope_parameters", "rope_scaling", "rope_theta"):
            kwargs.pop(name, None)
        kwargs.setdefault("image_token_index", 19)
        super().__init__(text_config=text_config, **kwargs)
        self.block_size = block_size
        self.canvas_length = block_size
        self.mask_token_id = mask_token_id
        self.confidence_threshold = confidence_threshold
        self.decode_policy = "nemotron_vlm"


class NemotronLabsDiffusionVLMTokenizerFast(PreTrainedTokenizerFast):
    pass
