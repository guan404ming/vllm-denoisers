import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("vllm")

from vllm import LLM, SamplingParams

from denoisers.models.sigma.config import SigmaModelConfig
from denoisers.models.sigma.model import NemotronLabsSigmaModel
from denoisers.models.sigma.params import SigmaSamplingParams
from denoisers.models.sigma.state import SigmaModelState, SigmaSampler


def test_sigma_reordered_rows_and_partial_block():
    batch = SimpleNamespace(
        num_reqs=3,
        req_ids=["b", "prompt", "a"],
        query_start_loc=torch.tensor([0, 3, 6, 8]),
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
    assert [(rid, start, width, prefix.tolist()) for rid, start, width, prefix in blocks] == [
        ("b", 0, 3, [[8, 9]]),
        ("a", 6, 2, [[4, 5, 6]]),
    ]
    sampler = SigmaSampler(SimpleNamespace(penalties_state=None), 3, 3, "cpu")
    out = sampler(torch.tensor([[-1], [-1], [-1], [1], [3]], dtype=torch.int32), batch)
    assert out.num_sampled.tolist() == [0, 0, 2]
    assert out.num_rejected.tolist() == [3, 0, 0]
    assert out.sampled_token_ids.tolist() == [[0, 0, 0], [0, 0, 0], [1, 3, 0]]


def test_sigma_resumes_by_request_and_discards_cancelled_state(monkeypatch):
    started = []

    class Block:
        def __init__(self, model, prefix, embeddings):
            self.token = prefix[0, -1]
            started.append(int(self.token))

        def run(self):
            yield None
            yield self.token.repeat(3)

    monkeypatch.setattr("denoisers.models.sigma.model.SigmaBlock", Block)
    model = SimpleNamespace(pending={}, embeddings=torch.zeros(1), model=None)
    state = object.__new__(SigmaModelState)
    state.model = model
    state.rope_state = None
    state.prompt_embeds_state = None
    positions = torch.arange(4)
    a = ("a", 0, 2, torch.tensor([[7]]))
    b = ("b", 2, 2, torch.tensor([[8]]))
    forward = NemotronLabsSigmaModel.forward
    assert forward(model, None, positions, [a, b]).flatten().tolist() == [-1] * 4
    SigmaModelState.remove_request(state, "b")
    replacement = ("b", 0, 2, torch.tensor([[9]]))
    moved = ("a", 2, 1, a[3])
    assert forward(model, None, positions, [replacement, moved]).flatten().tolist() == [
        -1,
        -1,
        7,
        -1,
    ]
    assert started == [7, 8, 9]
    assert list(model.pending) == ["b"]


def test_sigma_base_loads_exported_schedule():
    config = SimpleNamespace(
        hf_config=SimpleNamespace(block_size=32),
        logits_processors=None,
        try_get_generation_config=dict,
    )
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
