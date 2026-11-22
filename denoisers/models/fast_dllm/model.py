from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM

from denoisers.models.mixin import BlockDiffusionModel


class FastDLLMQwenForCausalLM(BlockDiffusionModel, Qwen2ForCausalLM):
    pass
