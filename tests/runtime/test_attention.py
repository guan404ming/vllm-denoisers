import pytest
import torch

pytest.importorskip("vllm")

from vllm.vllm_flash_attn import flash_attn_varlen_func

from denoisers.runtime.attention import tree_mask

BLOCK, PAGE, HEADS, KV_HEADS, DIM = 4, 16, 8, 2, 128


def allowed(code: int, ctx: int, length: int) -> torch.Tensor:
    query = torch.arange(length)[:, None]
    key = torch.arange(ctx + length)[None, :]
    span = code * BLOCK
    fills = span - BLOCK
    low = (ctx + query) - (ctx + query) % BLOCK
    if span < 1:
        return (key < low + BLOCK) | (span + BLOCK < 1)
    own = (key >= low) & (key < low + BLOCK)
    chain = (2 * query < fills) & (key < low)
    tail = (2 * query >= fills) & (query < fills + BLOCK) & (2 * key < 2 * low - fills)
    return (key < ctx) | own | chain | tail


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_tree_mask_matches_masked_dense_attention():
    gen = torch.Generator().manual_seed(0)
    codes = [3, 2, 1, 0, -1]
    lengths = [3 * BLOCK, 2 * BLOCK, 2 * BLOCK, 2 * BLOCK, 6 * BLOCK]
    ctxs = [BLOCK * int(n) for n in torch.randint(1, 6, (len(codes),), generator=gen)]
    seqs = [ctx + length for ctx, length in zip(ctxs, lengths)]
    pages = [-(-seq // PAGE) for seq in seqs]
    order = torch.randperm(sum(pages), generator=gen)
    table = torch.zeros(len(seqs), max(pages), dtype=torch.int32)
    for row, count in enumerate(pages):
        table[row, :count] = order[sum(pages[:row]) : sum(pages[: row + 1])]
    table = table.cuda()
    k = torch.randn(sum(pages), PAGE, KV_HEADS, DIM, generator=gen).bfloat16().cuda()
    v = torch.randn(sum(pages), PAGE, KV_HEADS, DIM, generator=gen).bfloat16().cuda()
    q = torch.randn(sum(lengths), HEADS, DIM, generator=gen).bfloat16().cuda()
    starts = [0, *torch.tensor(lengths).cumsum(0).tolist()]
    out = torch.empty_like(q)
    flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        out=out,
        max_seqlen_q=max(lengths),
        cu_seqlens_q=torch.tensor(starts, dtype=torch.int32, device="cuda"),
        max_seqlen_k=max(seqs),
        seqused_k=torch.tensor(seqs, dtype=torch.int32, device="cuda"),
        softmax_scale=DIM**-0.5,
        causal=False,
        block_table=table,
        fa_version=4,
        mask_mod=tree_mask(BLOCK),
        aux_tensors=[torch.tensor(codes, dtype=torch.int32, device="cuda")],
    )
    for row, (code, ctx, length) in enumerate(zip(codes, ctxs, lengths)):
        keys = torch.arange(ctx + length, device="cuda")
        slots = table[row, keys // PAGE].long()
        key = k[slots, keys % PAGE].float().repeat_interleave(HEADS // KV_HEADS, 1)
        value = v[slots, keys % PAGE].float().repeat_interleave(HEADS // KV_HEADS, 1)
        rows = slice(starts[row], starts[row + 1])
        scores = torch.einsum("qhd,khd->hqk", q[rows].float(), key) * DIM**-0.5
        scores = scores.masked_fill(~allowed(code, ctx, length).cuda(), float("-inf"))
        want = torch.einsum("hqk,khd->qhd", scores.softmax(-1), value)
        torch.testing.assert_close(out[rows].float(), want, atol=1e-2, rtol=0)
