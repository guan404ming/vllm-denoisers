import pytest
import torch

from denoisers.models.fp8_head import Fp8Head


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_fp8_head_tracks_bf16_logits():
    gen = torch.Generator().manual_seed(0)
    weight = (torch.randn(4096, 256, generator=gen) * 0.05).to(torch.bfloat16).cuda()
    x = torch.randn(64, 256, generator=gen).to(torch.bfloat16).cuda()
    want = torch.nn.functional.linear(x, weight).float()
    got = Fp8Head(weight).apply(None, x).float()
    assert (got - want).norm() / want.norm() < 0.05
    assert (got.argmax(-1) == want.argmax(-1)).float().mean() > 0.9
    assert Fp8Head(weight).apply(None, x[:0]).shape == (0, 4096)
