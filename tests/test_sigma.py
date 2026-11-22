import os
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

pytest.importorskip("vllm")

from vllm import LLM, SamplingParams

from denoisers.models.sigma.config import SigmaModelConfig
from denoisers.models.sigma.model import NemotronLabsSigmaModel
from denoisers.models.sigma.params import SigmaSamplingParams
from denoisers.models.sigma.state import SigmaModelState, SigmaSampler


@pytest.mark.parametrize("steps", [None, 2, 4, 8, 16])
def test_sigma_uses_checkpoint_recipe(steps):
    prefix = torch.tensor([[7, 8]])
    tokens = torch.tensor([[7, 8, 4, 3]])
    model = Mock()
    model.config.native_family = "phased_dmd" if steps else None
    model.config.native_nfe = steps
    model.generation_config.decoding = "blockwise_diffusion"
    model.generate.return_value = tokens, 0
    adapter = SimpleNamespace(model=model, block=32)

    result = NemotronLabsSigmaModel.generate_block(adapter, prefix, 2)

    model.generate.assert_called_once_with(
        prefix,
        max_new_tokens=2,
        block_length=32,
        steps=steps or 32,
        sampler="fresh_x0" if steps else "ddim",
        time_schedule="log_snr",
        guidance_scale=1.0 if steps else 3.0,
        score_temp=1.0 if steps else 0.3,
    )
    assert result.tolist() == [4, 3]


def test_sigma_speculative_uses_ar_verifier():
    model = Mock()
    model.generation_config = SimpleNamespace(
        decoding="self_speculative", draft_width=32, diffusion_calls_per_round=1
    )
    model.generate_speculative.return_value = torch.tensor([[7, 4, 3]])
    prefix = torch.tensor([[7]])
    result = NemotronLabsSigmaModel.generate_block(SimpleNamespace(model=model), prefix, 2)
    model.generate_speculative.assert_called_once_with(
        prefix, max_new_tokens=2, draft_width=32, diffusion_calls_per_round=1
    )
    assert result.tolist() == [4, 3]


def test_sigma_reordered_rows_and_partial_block():
    batch = SimpleNamespace(
        num_reqs=3,
        num_draft_tokens=5,
        idx_mapping_np=np.array([2, 0, 1]),
        query_start_loc_np=np.array([0, 3, 6, 8]),
        num_computed_tokens_np=np.array([2, 0, 3]),
        cu_num_logits_np=np.array([0, 3, 3, 5]),
        cu_num_logits=torch.tensor([0, 3, 3, 5]),
        expanded_local_pos=torch.tensor([0, 1, 2, 0, 1]),
    )
    states = SimpleNamespace(all_token_ids=SimpleNamespace(gpu=torch.arange(12).reshape(3, 4)))
    blocks = SigmaModelState.prepare_inputs(None, batch, states)["blocks"]
    assert [(start, width, prefix.tolist()) for start, width, prefix in blocks] == [
        (0, 3, [[8, 9]]),
        (6, 2, [[4, 5, 6]]),
    ]
    sampler = SigmaSampler(SimpleNamespace(penalties_state=None), 3)
    out = sampler(torch.tensor([[2], [4], [6], [1], [3]]), batch)
    assert out.num_sampled.tolist() == [3, 0, 2]
    assert out.sampled_token_ids.tolist() == [[2, 4, 6], [0, 0, 0], [1, 3, 0]]


def test_sigma_base_loads_exported_schedule():
    config = SimpleNamespace(hf_config=SimpleNamespace(block_size=32), logits_processors=None)
    SigmaModelConfig.verify_and_update_model_config(config)
    assert config.hf_config.schedule_parameterization == "projected_bernstein"
    assert config.hf_config.canvas_length == 32


def test_sigma_rejects_logprobs_at_admission():
    with pytest.raises(ValueError, match="logprobs"):
        SigmaSamplingParams.validate_params(SamplingParams(logprobs=0))


@pytest.mark.skipif(
    not os.getenv("DENOISERS_SIGMA_MODEL"), reason="needs a Sigma checkpoint on CUDA"
)
def test_sigma_generates_across_blocks():
    llm = LLM(
        model=os.environ["DENOISERS_SIGMA_MODEL"],
        trust_remote_code=True,
        enforce_eager=True,
        async_scheduling=False,
        enable_prefix_caching=False,
        max_model_len=128,
        max_num_seqs=2,
        max_num_batched_tokens=128,
        gpu_memory_utilization=float(os.getenv("DENOISERS_GPU_MEMORY_UTILIZATION", "0.9")),
    )
    prompts = ["The capital of France is", "One plus one is"]
    results = llm.generate(prompts, SamplingParams(max_tokens=36, ignore_eos=True))
    for result in results:
        ids = result.outputs[0].token_ids
        assert len(ids) == 36
        assert len(set(ids)) > 1
