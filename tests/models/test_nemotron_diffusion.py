from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import AutoConfig

pytest.importorskip("vllm")

from denoisers.canvas.state import CanvasState
from denoisers.models.families import register_all
from denoisers.policies.decode import NemotronDiffusion
from denoisers.runtime.decoders import Decoders
from denoisers.runtime.model_state import BlockDiffusionModelState
from denoisers.runtime.sampler import BlockDiffusionSampler


def test_nemotron_seeds_from_prompt_then_committed_block():
    register_all()
    config = AutoConfig.for_model("nemotron_labs_diffusion", block_size=4)
    assert config.canvas_length == 4
    assert config.decode_policy == "nemotron_diffusion"
    policy = NemotronDiffusion.from_config(config, None)
    assert policy.prefill_seed(torch.eye(4)[[2, 0]]).tolist() == [2, 0]
    logits = torch.eye(4)[[0, 3, 1, 2]].view(2, 2, 4)
    assert policy.next_seed(logits).tolist() == [3, 2]


@pytest.mark.parametrize("threshold,expected", [(0.9, [8, 1, 2, 100]), (None, [8, 1, 100, 100])])
def test_nemotron_threshold_and_fallback(threshold, expected):
    policy = NemotronDiffusion.from_config(
        SimpleNamespace(mask_token_id=100, confidence_threshold=threshold), None
    )
    canvas = torch.tensor([[8, 100, 100, 100], [8, 100, 100, 100]])
    tokens = torch.arange(4).expand_as(canvas)
    scores = torch.tensor([[1.0, 0.99, 0.9, 0.1], [1.0, 0.2, 0.3, 0.4]])
    after, done = policy.move(canvas, None, None, None, tokens, scores)
    assert after.tolist() == [expected, [8, 100, 100, 3]]
    assert done.tolist() == [False, False]


def test_mixed_prefill_seeds_only_finished_requests():
    config = SimpleNamespace(mask_token_id=100, confidence_threshold=0.9)
    policy = NemotronDiffusion.from_config(config, None)
    state = CanvasState(5, 4, 100, "cpu")
    decoders = Decoders(policy, lambda cls: cls.from_config(config, None), state, 5)
    decoders.assign(1, SimpleNamespace(extra_args={"decode_policy": "nemotron_vlm"}))
    batch = SimpleNamespace(
        has_prefill=True,
        num_tokens=9,
        num_tokens_after_padding=9,
        num_reqs=5,
        positions=torch.arange(9),
        logits_indices=torch.tensor([0, 1]),
        idx_mapping_np=np.array([3, 4, 0, 1, 2]),
        query_start_loc_np=np.array([0, 2, 4, 5, 8, 9]),
        is_prefilling_np=np.array([False, True, True, True, True]),
        num_computed_prefill_tokens_np=np.array([2, 3, 0, 0, 0]),
        num_scheduled_tokens=np.array([2, 2, 1, 3, 1]),
        prefill_len_np=np.array([2, 5, 4, 3, 1]),
    )
    runtime = object.__new__(BlockDiffusionModelState)
    runtime.device = torch.device("cpu")
    runtime.model = SimpleNamespace(compute_logits=torch.nn.Identity())
    runtime.decoders = decoders
    runtime.state = state
    runtime._seed_slots = []
    runtime._fill_positions = lambda batch: batch.positions
    runtime._inputs_embeds = None

    runtime.prepare_inputs(batch, None)
    hidden = torch.eye(9)
    decode_hidden = runtime.compute_sample_logits(hidden[batch.logits_indices])
    assert torch.equal(decode_hidden, hidden[:2])

    sampler = object.__new__(BlockDiffusionSampler)
    sampler.state = state
    sampler.decoders = decoders
    sampler.aligned = False
    sampler._need_event = SimpleNamespace(synchronize=lambda: None)
    sampler._need = sampler._need_host = torch.zeros(5)
    sampler._commit_host = torch.zeros(5, dtype=torch.bool)
    sampler._stash = {}
    sampler.kv = None
    drafts = torch.full_like(state.canvas, -1)
    sampler.req_states = SimpleNamespace(
        all_token_ids=SimpleNamespace(gpu=torch.zeros(5, 5)), draft_tokens=drafts
    )
    sampler._finish_prefills(batch)
    assert drafts[:, 0].tolist() == [-1, 100, 8, -1, 3]
    assert batch.logits_indices.tolist() == [0, 1, 3, 8]

    batch.has_prefill = False
    batch.logits_indices = torch.tensor([0, 1])
    runtime.prepare_inputs(batch, None)
    assert runtime.compute_sample_logits(hidden[:2]).shape == (2, 9)
