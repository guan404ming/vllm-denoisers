import os
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from vllm import LLM, SamplingParams

from denoisers.models.sigma.config import SigmaModelConfig


def updated(decoding=None, **fields):
    config = SimpleNamespace(
        hf_config=SimpleNamespace(block_size=32, **fields),
        try_get_generation_config=lambda: {"decoding": decoding},
        override_generation_config={},
    )
    SigmaModelConfig.verify_and_update_model_config(config)
    return config.hf_config


def test_sigma_base_uses_guided_latent_policy():
    config = updated()
    assert config.decode_policy == "sigma" and config.canvas_length == 32
    assert (config.denoise_steps, config.guidance_scale, config.score_temperature) == (32, 3.0, 0.3)


def test_sigma_distilled_uses_checkpoint_step_count():
    config = updated("blockwise_diffusion", native_nfe=4)
    assert (config.denoise_steps, config.guidance_scale, config.score_temperature) == (4, 1.0, 1.0)


def test_sigma_self_speculative_runs_autoregressively():
    config = SimpleNamespace(
        hf_config=SimpleNamespace(block_size=32, native_nfe=2),
        try_get_generation_config=lambda: {"decoding": "self_speculative"},
        override_generation_config={"max_tokens": 9},
    )
    SigmaModelConfig.verify_and_update_model_config(config)
    assert config.hf_config.decode_policy is None
    assert not hasattr(config.hf_config, "canvas_length")
    assert config.override_generation_config == {"temperature": 0.0, "max_tokens": 9}


@pytest.mark.skipif(
    not os.getenv("DENOISERS_SIGMA_MODEL"), reason="needs a Sigma checkpoint on CUDA"
)
def test_sigma_generates_across_blocks():
    llm = LLM(
        model=os.environ["DENOISERS_SIGMA_MODEL"],
        trust_remote_code=True,
        max_model_len=128,
        max_num_seqs=2,
        gpu_memory_utilization=float(os.getenv("DENOISERS_GPU_MEMORY_UTILIZATION", "0.9")),
    )
    prompts = ["The capital of France is", "One plus one is"]
    results = llm.generate(prompts, SamplingParams(max_tokens=36, ignore_eos=True))
    for result in results:
        ids = result.outputs[0].token_ids
        assert len(ids) == 36
        assert len(set(ids)) > 1
