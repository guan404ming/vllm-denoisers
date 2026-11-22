import pytest

from denoisers.models.dream.config import DreamConfig
from denoisers.models.fast_dllm.config import FastDLLMQwenConfig
from denoisers.policies import POLICIES, DecodePolicy, LowConfidence, register_policy


def test_register_policy_adds_subclass():
    @register_policy("test_low_confidence")
    class Custom(LowConfidence):
        pass

    assert POLICIES.pop("test_low_confidence") is Custom


def test_register_policy_rejects_other_classes():
    with pytest.raises(ValueError):
        register_policy("test_plain")(object)
    assert "test_plain" not in POLICIES


def test_policy_defaults_need_no_hooks():
    policy = LowConfidence(mask_id=0, block_length=4, steps=4, threshold=0.9)
    assert isinstance(policy, DecodePolicy)
    assert policy.next_seed(None) is None
    assert policy.sparse_logits and policy.stop_exit and not policy.shifted


def test_decode_config_verify_fills_denoise_steps():
    config = DreamConfig()
    config.denoise_steps = None
    config.verify(None)
    assert config.denoise_steps == config.canvas_length


def test_family_verify_checks_its_own_fields():
    with pytest.raises(ValueError):
        FastDLLMQwenConfig(canvas_length=16).verify(None)
