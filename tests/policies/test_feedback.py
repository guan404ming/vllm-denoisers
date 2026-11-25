from types import SimpleNamespace

import torch
from support import make_rows

from denoisers.ops.embedding import ExpectedEmbedding
from denoisers.policies.feedback import SelfConditioning, Smooth

MASK = 5
VOCAB = 6
HIDDEN = 3


def test_smooth_mixes_expected_embeddings_into_masked_cells():
    weight = torch.randn(VOCAB, HIDDEN, generator=torch.Generator().manual_seed(0))
    smooth = Smooth(ExpectedEmbedding(weight), MASK, weight=0.3, initial=0.1, growth=0.1)
    canvas = torch.tensor([[1, MASK, MASK, 2], [MASK, MASK, 3, 4]])
    before = make_rows(canvas, step=[1, 5])
    before.committing = torch.tensor([False, True])
    after = make_rows(canvas)
    have = torch.tensor([[True, True, False, True]] * 2)
    logits = torch.randn(2, 4, VOCAB, generator=torch.Generator().manual_seed(1))
    soft = smooth.soft(logits, have, before, after)
    expected = 0.2 * torch.softmax(logits[0, 1], -1) @ weight
    assert torch.allclose(soft[0, 1], expected, atol=1e-6)
    assert soft[0, [0, 2, 3]].abs().sum() == 0
    assert soft[1].abs().sum() == 0
    embeds = torch.ones(4, HIDDEN)
    assert torch.equal(smooth.condition(embeds, soft[0]), embeds + soft[0])


def test_self_conditioning_drops_cells_without_logits():
    model = SimpleNamespace(input_embedding=lambda: torch.nn.Embedding(VOCAB, HIDDEN))
    policy = SelfConditioning(model)
    after = make_rows(torch.tensor([[1, MASK, MASK, 2]]))
    after.soft = torch.ones(1, 4, HIDDEN)
    have = torch.tensor([[True, False, True, True]])
    soft = policy.soft(None, have, after, after)
    assert soft[0, :, 0].tolist() == [1.0, 0.0, 1.0, 1.0]
