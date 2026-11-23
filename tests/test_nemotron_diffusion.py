from types import SimpleNamespace

import pytest
import torch
from transformers import AutoConfig

pytest.importorskip("vllm")

from denoisers.models.families import register_all
from denoisers.policies.decode import NemotronDiffusion


def test_nemotron_seeds_from_prompt_then_committed_block():
    register_all()
    config = AutoConfig.for_model("nemotron_labs_diffusion", block_size=4)
    assert config.canvas_length == 4
    assert config.decode_policy == "nemotron_diffusion"
    model = SimpleNamespace(prompt_hidden=torch.eye(4), compute_logits=torch.nn.Identity())
    policy = NemotronDiffusion.from_config(config, model)
    slots = torch.tensor([2, 0])
    canvas = torch.full((2, 4), config.mask_token_id)
    assert policy.init_canvas(canvas, slots, first=True).tolist() == [
        [2, 100, 100, 100],
        [0, 100, 100, 100],
    ]
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
