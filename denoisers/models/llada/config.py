from transformers import LlamaConfig

from denoisers.models.config import DecodeConfig


class LLaDAConfig(DecodeConfig, LlamaConfig):
    model_type = "llada"
    attribute_map = {
        "hidden_size": "d_model",
        "num_hidden_layers": "n_layers",
        "num_attention_heads": "n_heads",
        "num_key_value_heads": "n_kv_heads",
        "intermediate_size": "mlp_hidden_size",
        "max_position_embeddings": "max_sequence_length",
        "tie_word_embeddings": "weight_tying",
        "hidden_act": "activation_type",
    }

    def __init__(
        self,
        canvas_length: int = 256,
        sub_block_length: int = 32,
        mask_token_id: int = 126336,
        decode_policy: str = "llada",
        temperature: float = 0.0,
        denoise_steps: int | None = None,
        confidence_threshold: float = 2.0,
        **kwargs,
    ):
        self.canvas_length = canvas_length
        self.sub_block_length = sub_block_length
        self.mask_token_id = mask_token_id
        self.decode_policy = decode_policy
        self.temperature = temperature
        self.denoise_steps = denoise_steps
        self.confidence_threshold = confidence_threshold
        super().__init__(**kwargs)
