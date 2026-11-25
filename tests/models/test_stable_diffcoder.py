from types import SimpleNamespace

import pytest
from transformers import AutoConfig, LlamaConfig

pytest.importorskip("vllm")

from vllm.model_executor.models.config import MODELS_CONFIG_MAP

from denoisers.models.families import register_all
from denoisers.policies.decode import LowConfidence


def test_stable_diffcoder_keeps_llama_registration_and_sets_decode_defaults():
    register_all()
    config = AutoConfig.for_model("llama", architectures=["StableDiffcoderForCausalLM"])
    assert type(config) is LlamaConfig
    verifier = MODELS_CONFIG_MAP[config.architectures[0]]
    verifier.verify_and_update_model_config(SimpleNamespace(hf_config=config))
    policy = LowConfidence.from_config(config, None)
    assert config.decode_policy == "low_confidence"
    assert config.canvas_length == 4
    assert policy.mask_id == 5
    assert policy.temperature == 0
    assert policy.threshold == 1
    assert policy.schedule.tolist() == [1, 1, 1, 1]
    assert not hasattr(AutoConfig.for_model("llama"), "canvas_length")
