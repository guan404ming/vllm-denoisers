from transformers import AutoConfig, AutoTokenizer
from vllm import ModelRegistry
from vllm.model_executor.models.config import MODELS_CONFIG_MAP

from denoisers.models.diffusion_gemma.model import DiffusionGemmaForBlockDiffusionConfig
from denoisers.models.dream.config import DreamConfig
from denoisers.models.dream.tokenizer import DreamTokenizer
from denoisers.models.fast_dllm.config import FastDLLMQwenConfig
from denoisers.models.llada.config import LLaDAConfig
from denoisers.models.llada2.config import LLaDA2MoeConfig
from denoisers.models.llada2.tokenizer import CustomFastTokenizer
from denoisers.models.sdar.config import SDARConfig, SDARMoeConfig
from denoisers.models.sigma.config import SigmaModelConfig
from denoisers.models.stable_diffcoder.model import StableDiffcoderModelConfig
from denoisers.models.verify import BlockDiffusionModelConfig

FAMILIES = (
    ("LLaDA2MoeModelLM", "llada2.model:LLaDA2MoeModelLM", LLaDA2MoeConfig, CustomFastTokenizer),
    ("SDARForCausalLM", "sdar.model:SDARForCausalLM", SDARConfig, None),
    ("SDARMoeForCausalLM", "sdar.model:SDARMoeForCausalLM", SDARMoeConfig, None),
    (
        "Fast_dLLM_QwenForCausalLM",
        "fast_dllm.model:FastDLLMQwenForCausalLM",
        FastDLLMQwenConfig,
        None,
    ),
    ("DreamModel", "dream.model:DreamModel", DreamConfig, DreamTokenizer),
    ("LLaDAModelLM", "llada.model:LLaDAModelLM", LLaDAConfig, None),
)


def register_family(
    architecture: str,
    model: str,
    config=None,
    tokenizer=None,
    verifier=BlockDiffusionModelConfig,
) -> None:
    if config is not None:
        AutoConfig.register(config.model_type, config, exist_ok=True)
    if tokenizer is not None:
        AutoTokenizer.register(config, fast_tokenizer_class=tokenizer, exist_ok=True)
    ModelRegistry.register_model(architecture, model)
    MODELS_CONFIG_MAP[architecture] = verifier


def register_all() -> None:
    for architecture, model, config, tokenizer in FAMILIES:
        register_family(architecture, f"denoisers.models.{model}", config, tokenizer)
    register_family(
        "NemotronLabsSigmaModel",
        "denoisers.models.sigma.model:NemotronLabsSigmaModel",
        verifier=SigmaModelConfig,
    )
    register_family(
        "StableDiffcoderForCausalLM",
        "denoisers.models.stable_diffcoder.model:StableDiffcoderForCausalLM",
        verifier=StableDiffcoderModelConfig,
    )
    register_family(
        "DiffusionGemmaForBlockDiffusion",
        "denoisers.models.diffusion_gemma.model:DiffusionGemma",
        verifier=DiffusionGemmaForBlockDiffusionConfig,
    )
