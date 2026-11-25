from vllm.model_executor.models.qwen3 import Qwen3ForCausalLM
from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM

from denoisers.models.common.mixin import BlockDiffusionModel


class SDARForCausalLM(BlockDiffusionModel, Qwen3ForCausalLM):
    pass


class SDARMoeForCausalLM(BlockDiffusionModel, Qwen3MoeForCausalLM):
    pass
