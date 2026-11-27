import os
from unittest.mock import patch

import pytest
import torch
from transformers import AutoConfig, AutoModel

from denoisers.models.sigma.steps import SigmaBlock


@pytest.fixture(scope="module")
def sigma():
    path = os.getenv("DENOISERS_SIGMA_MODEL")
    if not path:
        pytest.skip("set DENOISERS_SIGMA_MODEL to a local Sigma checkpoint")
    config = AutoConfig.from_pretrained(path, trust_remote_code=True, local_files_only=True)
    config.vocab_size = 64
    config.hidden_size = 32
    config.intermediate_size = 64
    config.num_hidden_layers = 1
    config.num_attention_heads = 2
    config.num_key_value_heads = 1
    config.head_dim = 16
    config.block_size = 4
    config.include_ar_head = True
    config.schedule_parameterization = "projected_bernstein"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        return AutoModel.from_config(config, trust_remote_code=True).eval()


@pytest.mark.parametrize(
    "recipe,nfe",
    [
        ("ddim", 32),
        ("self_speculative", 2),
    ],
)
@torch.inference_mode()
def test_sigma_matches_reference_at_each_step(sigma, monkeypatch, recipe, nfe):
    speculative = recipe == "self_speculative"
    monkeypatch.setattr(sigma.config, "native_nfe", nfe, raising=False)
    monkeypatch.setattr(sigma.generation_config, "decoding", recipe, raising=False)
    monkeypatch.setattr(sigma.generation_config, "draft_width", 3, raising=False)
    monkeypatch.setattr(sigma.generation_config, "diffusion_calls_per_round", nfe, raising=False)
    prompt = torch.tensor([[5, 11, 3]])
    width = sigma.config.block_size
    with (
        torch.random.fork_rng(devices=[]),
        patch.object(sigma, "_denoise_block", wraps=sigma._denoise_block) as denoise,
    ):
        torch.manual_seed(17)
        if speculative:
            expected = sigma.generate_speculative(
                prompt, max_new_tokens=width, diffusion_calls_per_round=nfe
            )
        else:
            expected, _ = sigma.generate(
                prompt,
                max_new_tokens=width,
                block_length=width,
                steps=nfe,
                sampler=recipe,
                time_schedule="log_snr",
                guidance_scale=3.0,
                score_temp=0.3,
            )
        reference_calls = denoise.call_args_list.copy()
        denoise.reset_mock()
        torch.manual_seed(17)
        block = SigmaBlock(sigma, prompt, sigma.replaid_embedding_matrix().float())
        steps = block.run()
        previous_calls = 0
        for tokens in steps:
            assert denoise.call_count - previous_calls <= 1
            previous_calls = denoise.call_count
            if tokens is not None:
                assert torch.equal(tokens, expected[0, prompt.shape[1] :])
                break
        else:
            pytest.fail("block finished without emitting tokens")

        assert denoise.call_count == len(reference_calls)
        for actual, reference in zip(denoise.call_args_list, reference_calls, strict=True):
            torch.testing.assert_close(actual.args[0], reference.args[0], atol=0, rtol=0)
            torch.testing.assert_close(actual.args[1], reference.args[1], atol=0, rtol=0)
            torch.testing.assert_close(actual.args[6], reference.args[6], atol=0, rtol=0)
