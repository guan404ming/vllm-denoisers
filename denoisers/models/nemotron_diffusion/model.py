from vllm.model_executor.models.mistral import MistralForCausalLM
from vllm.model_executor.models.utils import WeightsMapper

from denoisers.models.common.mixin import BlockDiffusionModel


class NemotronLabsDiffusionModel(BlockDiffusionModel, MistralForCausalLM):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"encoder.": "model.", "diffusion_head.": "lm_head."}
    )

    def load_weights(self, weights):
        return super().load_weights(self.hf_to_vllm_mapper.apply(weights))
