from types import SimpleNamespace

import pytest
import torch

from denoisers.ops.lm_head import LM_HEAD_METHODS, Int4Head


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("name,error_limit", [("fp8", 0.05), ("int4", 0.1)])
def test_quantized_head_tracks_bf16_logits(name, error_limit):
    gen = torch.Generator().manual_seed(0)
    weight = (torch.randn(4096, 256, generator=gen) * 0.05).to(torch.bfloat16).cuda()
    x = torch.randn(2, 32, 256, generator=gen).to(torch.bfloat16).cuda()
    method = LM_HEAD_METHODS[name](weight)
    layer = SimpleNamespace(weight=weight)
    want = torch.nn.functional.linear(x, weight).float()
    got = method.apply(layer, x).float()
    assert (got - want).norm() / want.norm() < error_limit
    if name == "fp8":
        assert (got.argmax(-1) == want.argmax(-1)).float().mean() > 0.9
    assert method.apply(layer, x[:, :0]).shape == (2, 0, 4096)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_int4_head_matches_groupwise_projection_with_bias():
    codes = torch.arange(32, device="cuda") % 16
    scales = (torch.arange(24, device="cuda") % 8 + 1).view(8, 3, 1)
    offsets = torch.arange(8, device="cuda").view(8, 1, 1)
    weight = (codes * scales - offsets).reshape(8, 96).bfloat16()
    weight[0] = 3
    x = torch.eye(96, dtype=weight.dtype, device=weight.device)
    bias = torch.arange(8, dtype=weight.dtype, device=weight.device)
    method = Int4Head(weight)
    assert method.weight.nbytes == weight.numel() // 2
    want = torch.nn.functional.linear(x, weight, bias)
    got = method.apply(SimpleNamespace(weight=weight), x, bias)
    torch.testing.assert_close(got, want, rtol=0, atol=0)
