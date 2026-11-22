import pytest

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
