import pytest

from denoisers.models.dream.config import DreamConfig
from denoisers.models.fast_dllm.config import FastDLLMQwenConfig
from denoisers.policies.decode import DecodePolicy, LowConfidence
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


def test_decode_config_verify_fills_denoise_steps():
    config = DreamConfig()
    config.denoise_steps = None
    config.verify(None)
    assert config.denoise_steps == config.canvas_length


def test_family_verify_checks_its_own_fields():
    with pytest.raises(ValueError):
        FastDLLMQwenConfig(canvas_length=16).verify(None)
