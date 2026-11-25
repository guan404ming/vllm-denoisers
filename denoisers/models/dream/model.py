from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM

from denoisers.models.common.mixin import BlockDiffusionModel


class DreamModel(BlockDiffusionModel, Qwen2ForCausalLM):
    pass
