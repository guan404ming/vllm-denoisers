from transformers import LlamaConfig, Qwen3MoeConfig

from denoisers.models.common.config import DecodeConfig


class LLaDADecodeConfig(DecodeConfig):
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


class LLaDAMoEConfig(LLaDADecodeConfig, Qwen3MoeConfig):
    model_type = "llada"

    def __init__(
        self,
        mask_token_id: int = 156895,
        expert_intermediate_size: int = 1024,
        norm_topk_prob: bool | None = False,
        shared_expert_intermediate_size: int | None = 0,
        **kwargs,
    ):
        super().__init__(
            mask_token_id=mask_token_id,
            moe_intermediate_size=expert_intermediate_size,
            norm_topk_prob=bool(norm_topk_prob),
            shared_expert_intermediate_size=shared_expert_intermediate_size or 0,
            **kwargs,
        )


class LLaDAConfig(LLaDADecodeConfig, LlamaConfig):
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
    __init__ = LLaDADecodeConfig.__init__

    @classmethod
    def from_dict(cls, config_dict, **kwargs):
        if cls is LLaDAConfig and "LLaDAMoEModel" in config_dict.get("architectures", ()):
            return LLaDAMoEConfig.from_dict(config_dict, **kwargs)
        return super().from_dict(config_dict, **kwargs)
