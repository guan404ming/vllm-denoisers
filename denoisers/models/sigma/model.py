from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor.models.utils import AutoWeightsLoader, WeightsMapper

from denoisers.models.sigma.diffusion import NemotronLabsSigmaBlockDiffusion, use_sigma_rope


class NemotronLabsSigmaModel(LlamaForCausalLM):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"encoder.": "model.", "ar_head.": "lm_head."}
    )

    def __new__(cls, *, vllm_config, prefix=""):
        if vllm_config.model_config.hf_config.decode_policy == "sigma":
            return NemotronLabsSigmaBlockDiffusion(vllm_config=vllm_config, prefix=prefix)
        return super().__new__(cls)

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        use_sigma_rope(self)

    def load_weights(self, weights):
        loader = AutoWeightsLoader(
            self,
            ignore_unexpected_prefixes=[
                "diffusion_head.",
                "replaid_",
                "noise_schedule.",
                "gamma_bounds.",
            ],
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)
