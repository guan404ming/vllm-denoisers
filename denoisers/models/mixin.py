from denoisers.models.fp8_head import Fp8Head
from denoisers.policies.decode import registry
from denoisers.runtime.model_state import BlockDiffusionModelState


class BlockDiffusionModel:
    @staticmethod
    def get_model_state_cls():
        return BlockDiffusionModelState

    def build_policy(self):
        return registry.resolve(self.config.decode_policy).from_config(self.config, self)

    def input_embedding(self):
        return self.model.embed_tokens

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        if getattr(self.config, "fp8_lm_head", False):
            self.lm_head.quant_method = Fp8Head(self.lm_head.weight)
        return loaded
