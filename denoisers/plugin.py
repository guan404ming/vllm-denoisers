from transformers import AutoConfig, AutoTokenizer
from vllm.model_executor.models.config import MODELS_CONFIG_MAP

from denoisers.models.dream.config import DreamConfig
from denoisers.models.dream.model import DreamModelConfig
from denoisers.models.dream.tokenizer import DreamTokenizer
from denoisers.models.fast_dllm.config import FastDLLMQwenConfig
from denoisers.models.diffusion_gemma.model import DiffusionGemmaForBlockDiffusionConfig
from denoisers.models.fast_dllm.model import FastDLLMQwenForCausalLMConfig
from denoisers.models.llada.config import LLaDAConfig
from denoisers.models.llada.model import LLaDAModelLMConfig
from denoisers.models.llada2.config import LLaDA2MoeConfig
from denoisers.models.llada2.model import LLaDA2MoeModelLMConfig
from denoisers.models.llada2.tokenizer import CustomFastTokenizer
from denoisers.models.sdar.config import SDARConfig, SDARMoeConfig
from denoisers.models.sdar.model import SDARForCausalLMConfig
from vllm import ModelRegistry
from vllm.v1.worker.gpu.spec_decode.utils import DraftTokensHandler

from denoisers.canvas import prune

_set_draft_tokens = DraftTokensHandler.set_draft_tokens


def set_draft_tokens(self, input_batch, draft_tokens) -> None:
    _set_draft_tokens(self, input_batch, draft_tokens)
    if prune.provider is None:
        return
    for req_id, width in zip(input_batch.req_ids, prune.provider(input_batch).tolist()):
        drafts = self._draft_tokens_by_req[req_id]
        self._draft_tokens_by_req[req_id] = (
            min(drafts, width) if isinstance(drafts, int) else drafts[:width]
        )


def register() -> None:
    DraftTokensHandler.set_draft_tokens = set_draft_tokens
    AutoConfig.register("llada2_moe", LLaDA2MoeConfig, exist_ok=True)
    AutoTokenizer.register(
        LLaDA2MoeConfig, fast_tokenizer_class=CustomFastTokenizer, exist_ok=True
    )
    ModelRegistry.register_model(
        "LLaDA2MoeModelLM", "denoisers.models.llada2.model:LLaDA2MoeModelLM"
    )
    MODELS_CONFIG_MAP["LLaDA2MoeModelLM"] = LLaDA2MoeModelLMConfig
    AutoConfig.register("sdar", SDARConfig, exist_ok=True)
    AutoConfig.register("sdar_moe", SDARMoeConfig, exist_ok=True)
    ModelRegistry.register_model("SDARForCausalLM", "denoisers.models.sdar.model:SDARForCausalLM")
    ModelRegistry.register_model(
        "SDARMoeForCausalLM", "denoisers.models.sdar.model:SDARMoeForCausalLM"
    )
    MODELS_CONFIG_MAP["SDARForCausalLM"] = SDARForCausalLMConfig
    MODELS_CONFIG_MAP["SDARMoeForCausalLM"] = SDARForCausalLMConfig
    AutoConfig.register("Fast_dLLM_Qwen", FastDLLMQwenConfig, exist_ok=True)
    ModelRegistry.register_model(
        "Fast_dLLM_QwenForCausalLM", "denoisers.models.fast_dllm.model:FastDLLMQwenForCausalLM"
    )
    MODELS_CONFIG_MAP["Fast_dLLM_QwenForCausalLM"] = FastDLLMQwenForCausalLMConfig
    ModelRegistry.register_model(
        "DiffusionGemmaForBlockDiffusion",
        "denoisers.models.diffusion_gemma.model:DiffusionGemma",
    )
    MODELS_CONFIG_MAP["DiffusionGemmaForBlockDiffusion"] = DiffusionGemmaForBlockDiffusionConfig
    AutoConfig.register("Dream", DreamConfig, exist_ok=True)
    AutoTokenizer.register(DreamConfig, fast_tokenizer_class=DreamTokenizer, exist_ok=True)
    ModelRegistry.register_model("DreamModel", "denoisers.models.dream.model:DreamModel")
    MODELS_CONFIG_MAP["DreamModel"] = DreamModelConfig
    AutoConfig.register("llada", LLaDAConfig, exist_ok=True)
    ModelRegistry.register_model("LLaDAModelLM", "denoisers.models.llada.model:LLaDAModelLM")
    MODELS_CONFIG_MAP["LLaDAModelLM"] = LLaDAModelLMConfig
