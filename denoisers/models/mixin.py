from typing import Literal

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.model_executor.layers.vocab_parallel_embedding import UnquantizedEmbeddingMethod

from denoisers.models.lm_head import LM_HEAD_METHODS, QuantizedHead
from denoisers.policies.decode import registry
from denoisers.runtime.model_state import BlockDiffusionModelState


class BlockDiffusionModel:
    image_encoder_output: Literal["embeddings", "token_ids"] = "embeddings"

    @staticmethod
    def get_model_state_cls():
        return BlockDiffusionModelState

    def build_policy(self):
        return registry.resolve(self.config.decode_policy).from_config(self.config, self)

    def input_embedding(self):
        return self.model.embed_tokens

    def load_weights(self, weights):
        name = getattr(self.config, "lm_head_quantization", None)
        if name is None:
            return super().load_weights(weights)
        if name not in LM_HEAD_METHODS:
            raise ValueError(
                f"Unknown lm_head_quantization {name!r}. Choose from {sorted(LM_HEAD_METHODS)}."
            )
        if not isinstance(
            self.lm_head.quant_method,
            (UnquantizedEmbeddingMethod, UnquantizedLinearMethod, QuantizedHead),
        ):
            raise ValueError("lm_head_quantization requires an unquantized LM head.")
        loaded = super().load_weights(weights)
        self.lm_head.quant_method = LM_HEAD_METHODS[name](self.lm_head.weight.detach())
        return loaded
