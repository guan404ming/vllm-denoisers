from denoisers.canvas.model_state import BlockDiffusionModelState
from denoisers.models.fp8_head import Fp8Head
from denoisers.policies import build


class BlockDiffusionModel:
    @staticmethod
    def get_model_state_cls():
        return BlockDiffusionModelState

    def build_policy(self):
        return build(self.config.decode_policy, self.config)

    def load_weights(self, weights):
        loaded = super().load_weights(weights)
        if getattr(self.config, "fp8_lm_head", False):
            self.lm_head.quant_method = Fp8Head(self.lm_head.weight)
        return loaded
