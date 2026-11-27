from transformers import PretrainedConfig

from denoisers.models.common.config import DecodeConfig


class SDLLMConfig(DecodeConfig, PretrainedConfig):
    model_type = "sdllm"

    def __init__(
        self,
        vocab_size: int = 131072,
        hidden_size: int = 2176,
        intermediate_size: int = 8704,
        num_hidden_layers: int = 28,
        num_attention_heads: int = 17,
        max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        layer_norm_eps: float = 1e-5,
        tie_word_embeddings: bool = False,
        canvas_length: int = 256,
        sub_block_length: int = 32,
        mask_token_id: int = 3,
        decode_policy: str = "llada",
        temperature: float = 0.0,
        denoise_steps: int | None = None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.layer_norm_eps = layer_norm_eps
        self.canvas_length = canvas_length
        self.sub_block_length = sub_block_length
        self.mask_token_id = mask_token_id
        self.decode_policy = decode_policy
        self.temperature = temperature
        self.denoise_steps = denoise_steps
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
