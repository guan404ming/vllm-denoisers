import pytest

from denoisers.models.dream.config import DreamConfig
from denoisers.models.fast_dllm.config import FastDLLMQwenConfig
from denoisers.policies import decode, hold, inputs, kv
from denoisers.policies.decode import DecodePolicy, LowConfidence
from denoisers.policies.kv import Masked
from denoisers.registry import Registry


def test_registry_registers_subclasses_and_resolves_qualified_names():
    registry = Registry("thing", DecodePolicy)

    @registry.register("custom")
    class Custom(LowConfidence):
        pass

    assert registry.resolve("custom") is Custom
    qualified = "denoisers.policies.decode.low_confidence.LowConfidence"
    assert registry.resolve(qualified) is LowConfidence
    with pytest.raises(ValueError):
        registry.resolve("unknown")
    with pytest.raises(ValueError):
        registry.register("plain")(object)
    with pytest.raises(ValueError):
        registry.resolve("denoisers.policies.kv.masked.Masked")


def test_built_ins_register_on_import():
    assert set(decode.registry.entries) == {
        "low_confidence",
        "joint",
        "indel",
        "fast_dllm",
        "entropy_bound",
        "dream",
        "llada",
    }
    assert kv.registry.resolve("masked") is Masked
    assert set(kv.registry.entries) == {"masked", "window"}
    assert set(inputs.registry.entries) == {"self_conditioning", "smooth"}
    assert set(hold.registry.entries) == {"sjf", "aging", "random"}


def test_policy_defaults_need_no_hooks():
    policy = LowConfidence(mask_id=0, block_length=4, steps=4, threshold=0.9)
    assert isinstance(policy, DecodePolicy)
    assert policy.next_seed(None) is None
    assert policy.focus(None) is None
    assert policy.sparse_logits and policy.stop_exit and not policy.shifted


def test_decode_config_verify_fills_denoise_steps():
    config = DreamConfig()
    config.denoise_steps = None
    config.verify(None)
    assert config.denoise_steps == config.canvas_length


def test_family_verify_checks_its_own_fields():
    with pytest.raises(ValueError):
        FastDLLMQwenConfig(canvas_length=16).verify(None)
