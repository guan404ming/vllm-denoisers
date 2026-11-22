from dataclasses import dataclass

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

BLOCK = 4096
CANDIDATES = 256


@dataclass
class RowStats:
    x0: torch.Tensor
    p0: torch.Tensor
    entropy: torch.Tensor
    alt: torch.Tensor


if triton is not None:

    @triton.jit
    def _first_index(values, indices, V):
        top = tl.max(values, 0)
        return top, tl.min(tl.where(values == top, indices, V), 0)

    @triton.autotune(
        configs=[
            triton.Config({"BLOCK": 512}, num_warps=4),
            triton.Config({"BLOCK": 1024}, num_warps=4),
            triton.Config({"BLOCK": 1024}, num_warps=8),
            triton.Config({"BLOCK": 2048}, num_warps=8),
            triton.Config({"BLOCK": 2048}, num_warps=4),
            triton.Config({"BLOCK": 4096}, num_warps=8),
        ],
        key=["V"],
    )
    @triton.jit
    def _row_stats_kernel(
        logits_ptr,
        stride,
        active_ptr,
        x0_ptr,
        p0_ptr,
        entropy_ptr,
        alt_ptr,
        V,
        mask_id,
        ex0,
        ex1,
        ex2,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        if tl.load(active_ptr + row) == 0:
            tl.store(x0_ptr + row, 0)
            tl.store(p0_ptr + row, 0.0)
            tl.store(entropy_ptr + row, 0.0)
            tl.store(alt_ptr + row, 0)
            return
        base = logits_ptr + row.to(tl.int64) * stride
        lane = tl.arange(0, BLOCK)
        neg = float("-inf")
        m = tl.full([BLOCK], neg, tl.float32)
        z = tl.zeros([BLOCK], tl.float32)
        s = tl.zeros([BLOCK], tl.float32)
        best = tl.full([BLOCK], neg, tl.float32)
        best_i = tl.zeros([BLOCK], tl.int32)
        alt_best = tl.full([BLOCK], neg, tl.float32)
        alt_i = tl.zeros([BLOCK], tl.int32)
        for start in range(0, V, BLOCK):
            offs = start + lane
            x = tl.load(base + offs, mask=offs < V, other=neg).to(tl.float32)
            new_m = tl.maximum(m, x)
            safe = tl.where(new_m > neg, new_m, 0.0)
            alpha = tl.where(m > neg, tl.exp(m - safe), 0.0)
            shift = tl.where(z > 0, (m - safe) * z, 0.0)
            d = x - safe
            e = tl.exp(d)
            z = z * alpha + e
            s = (s + shift) * alpha + tl.where(e > 0, d * e, 0.0)
            m = new_m
            xm = tl.where(offs == mask_id, neg, x)
            take = xm > best
            best_i = tl.where(take, offs, best_i)
            best = tl.where(take, xm, best)
            xa = tl.where((offs == ex0) | (offs == ex1) | (offs == ex2), neg, xm)
            take = xa > alt_best
            alt_i = tl.where(take, offs, alt_i)
            alt_best = tl.where(take, xa, alt_best)
        top = tl.max(m, 0)
        w = tl.where(m > neg, tl.exp(m - top), 0.0)
        total = tl.sum(z * w, 0)
        moment = tl.sum((s + tl.where(z > 0, (m - top) * z, 0.0)) * w, 0)
        x0_value, x0 = _first_index(best, best_i, V)
        _, alt = _first_index(alt_best, alt_i, V)
        tl.store(x0_ptr + row, x0.to(tl.int64))
        tl.store(p0_ptr + row, tl.exp(x0_value - top) / total)
        tl.store(entropy_ptr + row, tl.log(total) - moment / total)
        tl.store(alt_ptr + row, alt.to(tl.int64))

    @triton.jit
    def _sample_kernel(
        logits_ptr,
        stride,
        active_ptr,
        x0_ptr,
        p0_ptr,
        plain_ptr,
        noise_ptr,
        inv,
        V,
        mask_id,
        NB: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        if tl.load(active_ptr + row) == 0:
            tl.store(x0_ptr + row, 0)
            tl.store(p0_ptr + row, 0.0)
            tl.store(plain_ptr + row, 0.0)
            return
        base = logits_ptr + row.to(tl.int64) * stride
        lane = tl.arange(0, BLOCK)
        block = tl.arange(0, NB)
        neg = float("-inf")
        m = neg
        z = 0.0
        z1 = 0.0
        shifts = tl.zeros([NB], tl.float32)
        sums = tl.zeros([NB], tl.float32)
        for start in range(0, V, BLOCK):
            offs = start + lane
            x = tl.load(base + offs, mask=offs < V, other=neg).to(tl.float32)
            new_m = tl.maximum(m, tl.max(x, 0))
            safe = tl.where(new_m > neg, new_m, 0.0)
            e = tl.exp((x - safe) * inv)
            z = z * tl.exp((m - safe) * inv) + tl.sum(e, 0)
            z1 = z1 * tl.exp(m - safe) + tl.sum(tl.exp(x - safe), 0)
            m = new_m
            here = block == start // BLOCK
            shifts = tl.where(here, safe, shifts)
            sums = tl.where(here, tl.sum(tl.where(offs == mask_id, 0.0, e), 0), sums)
        mass = sums * tl.exp((shifts - m) * inv)
        reach = tl.cumsum(mass, 0)
        target = tl.load(noise_ptr + row) * tl.sum(mass, 0)
        k = tl.min(tl.where((reach >= target) & (mass > 0), block, (V - 1) // BLOCK), 0)
        seen = tl.sum(tl.where(block == k, reach - mass, 0.0), 0)
        offs = k * BLOCK + lane
        x = tl.load(base + offs, mask=offs < V, other=neg).to(tl.float32)
        e = tl.where(offs == mask_id, 0.0, tl.exp((x - m) * inv))
        hit = tl.min(tl.where((seen + tl.cumsum(e, 0) >= target) & (e > 0), lane, BLOCK), 0)
        pick = k * BLOCK + tl.where(hit < BLOCK, hit, tl.argmax(e, 0))
        value = tl.load(base + pick).to(tl.float32)
        tl.store(x0_ptr + row, pick.to(tl.int64))
        tl.store(p0_ptr + row, tl.exp((value - m) * inv) / z)
        tl.store(plain_ptr + row, tl.exp(value - m) / z1)

    @triton.jit
    def _nucleus_kernel(
        logits_ptr,
        stride,
        active_ptr,
        ids_ptr,
        probs_ptr,
        inv,
        V,
        mask_id,
        NB: tl.constexpr,
        K: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        slot = tl.arange(0, K)
        bag = row.to(tl.int64) * K
        if tl.load(active_ptr + row) == 0:
            tl.store(ids_ptr + bag + slot, tl.zeros([K], tl.int64))
            tl.store(probs_ptr + bag + slot, (slot == 0).to(tl.float32))
            return
        base = logits_ptr + row.to(tl.int64) * stride
        lane = tl.arange(0, BLOCK)
        block = tl.arange(0, NB)
        neg = float("-inf")
        m = neg
        z = 0.0
        best = neg
        tops = tl.full([NB], neg, tl.float32)
        for start in range(0, V, BLOCK):
            offs = start + lane
            x = tl.load(base + offs, mask=offs < V, other=neg).to(tl.float32)
            new_m = tl.maximum(m, tl.max(x, 0))
            safe = tl.where(new_m > neg, new_m, 0.0)
            z = z * tl.exp((m - safe) * inv) + tl.sum(tl.exp((x - safe) * inv), 0)
            m = new_m
            top = tl.max(tl.where(offs == mask_id, neg, x), 0)
            best = tl.maximum(best, top)
            tops = tl.where(block == start // BLOCK, top, tops)
        cut = tl.minimum(m + (tl.log(z) - tl.log(K * 1.0)) / inv, best)
        count = 0
        for k in range(NB):
            if tl.max(tl.where(block == k, tops, neg), 0) >= cut:
                offs = k * BLOCK + lane
                x = tl.load(base + offs, mask=offs < V, other=neg).to(tl.float32)
                flag = ((x >= cut) & (offs != mask_id) & (offs < V)).to(tl.int32)
                pos = count + tl.cumsum(flag, 0) - flag
                keep = (flag > 0) & (pos < K)
                tl.store(ids_ptr + bag + pos, offs.to(tl.int64), mask=keep)
                tl.store(probs_ptr + bag + pos, tl.exp((x - m) * inv) / z, mask=keep)
                count += tl.sum(flag, 0)
        unused = slot >= count
        tl.store(ids_ptr + bag + slot, tl.zeros([K], tl.int64), mask=unused)
        tl.store(probs_ptr + bag + slot, tl.zeros([K], tl.float32), mask=unused)


def _flatten(
    logits: torch.Tensor, active: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = logits.reshape(-1, logits.shape[-1])
    if active is None:
        active = torch.ones(rows.shape[0], dtype=torch.bool, device=rows.device)
    return rows, active.reshape(-1)


def _reference(
    logits: torch.Tensor, mask_id: int, exclude: tuple[int, ...], active: torch.Tensor
) -> RowStats:
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    probs = log_probs.exp()
    candidates = probs.clone()
    candidates[:, mask_id] = 0
    x0 = candidates.argmax(dim=-1)
    candidates[:, list(exclude)] = 0
    alt = candidates.argmax(dim=-1)
    entropy = -(probs * log_probs.clamp(min=-1e4)).sum(dim=-1)
    p0 = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
    zero = torch.zeros_like(x0)
    return RowStats(
        torch.where(active, x0, zero),
        torch.where(active, p0, 0.0),
        torch.where(active, entropy, 0.0),
        torch.where(active, alt, zero),
    )


def row_stats(
    logits: torch.Tensor,
    mask_id: int,
    exclude: tuple[int, ...] = (),
    active: torch.Tensor | None = None,
) -> RowStats:
    shape = logits.shape[:-1]
    rows, active = _flatten(logits, active)
    exclude = (tuple(exclude) + (mask_id,) * 3)[:3]
    if rows.is_cuda and triton is not None:
        n, vocab = rows.shape
        x0 = torch.empty(n, dtype=torch.int64, device=rows.device)
        alt = torch.empty_like(x0)
        p0 = torch.empty(n, dtype=torch.float32, device=rows.device)
        entropy = torch.empty_like(p0)
        if n:
            _row_stats_kernel[(n,)](
                rows,
                rows.stride(0),
                active.to(torch.int8),
                x0,
                p0,
                entropy,
                alt,
                vocab,
                mask_id,
                *exclude,
            )
        stats = RowStats(x0, p0, entropy, alt)
    else:
        stats = _reference(rows, mask_id, exclude, active)
    return RowStats(*(t.view(shape) for t in (stats.x0, stats.p0, stats.entropy, stats.alt)))


def _sample_reference(
    rows: torch.Tensor,
    mask_id: int,
    temperature: float,
    active: torch.Tensor,
    noise: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = rows[active].float()
    probs = torch.softmax(logits / temperature, dim=-1)
    probs[:, mask_id] = 0
    reach = probs.cumsum(dim=-1)
    target = noise[active].unsqueeze(-1) * reach[:, -1:]
    chosen = ((reach < target) | (probs <= 0)).int().argmin(dim=-1, keepdim=True)
    x0 = torch.zeros(rows.shape[0], dtype=torch.int64, device=rows.device)
    p0 = torch.zeros(rows.shape[0], dtype=torch.float32, device=rows.device)
    plain = torch.zeros_like(p0)
    x0[active] = chosen.squeeze(-1)
    p0[active] = probs.gather(-1, chosen).squeeze(-1)
    plain[active] = torch.softmax(logits, dim=-1).gather(-1, chosen).squeeze(-1)
    return x0, p0, plain


def sample_stats(
    logits: torch.Tensor,
    mask_id: int,
    temperature: float,
    active: torch.Tensor | None = None,
    noise: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shape = logits.shape[:-1]
    rows, active = _flatten(logits, active)
    if noise is None:
        noise = torch.rand(rows.shape[0], device=rows.device)
    noise = noise.reshape(-1).float()
    if rows.is_cuda and triton is not None:
        n, vocab = rows.shape
        x0 = torch.empty(n, dtype=torch.int64, device=rows.device)
        p0 = torch.empty(n, dtype=torch.float32, device=rows.device)
        plain = torch.empty_like(p0)
        if n:
            _sample_kernel[(n,)](
                rows,
                rows.stride(0),
                active.to(torch.int8),
                x0,
                p0,
                plain,
                noise,
                1.0 / temperature,
                vocab,
                mask_id,
                NB=triton.next_power_of_2(triton.cdiv(vocab, BLOCK)),
                BLOCK=BLOCK,
                num_warps=8,
            )
    else:
        x0, p0, plain = _sample_reference(rows, mask_id, temperature, active, noise)
    return x0.view(shape), p0.view(shape), plain.view(shape)


def _nucleus_reference(
    rows: torch.Tensor, mask_id: int, temperature: float, active: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    probs = torch.softmax(rows.float() / temperature, dim=-1)
    probs[:, mask_id] = 0
    cut = probs.amax(dim=-1, keepdim=True).clamp(max=1 / CANDIDATES)
    probs = torch.where(probs >= cut, probs, 0.0)
    probs, ids = probs.topk(min(CANDIDATES, probs.shape[-1]), dim=-1)
    first = (torch.arange(probs.shape[-1], device=rows.device) == 0).float()
    keep = active.unsqueeze(-1)
    return torch.where(keep, ids, 0), torch.where(keep, probs, first)


def nucleus_bag(
    logits: torch.Tensor,
    mask_id: int,
    temperature: float,
    active: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = logits.shape[:-1]
    rows, active = _flatten(logits, active)
    if rows.is_cuda and triton is not None:
        n, vocab = rows.shape
        ids = torch.empty(n, CANDIDATES, dtype=torch.int64, device=rows.device)
        probs = torch.empty(n, CANDIDATES, dtype=torch.float32, device=rows.device)
        if n:
            _nucleus_kernel[(n,)](
                rows,
                rows.stride(0),
                active.to(torch.int8),
                ids,
                probs,
                1.0 / temperature,
                vocab,
                mask_id,
                NB=triton.next_power_of_2(triton.cdiv(vocab, BLOCK)),
                K=CANDIDATES,
                BLOCK=BLOCK,
                num_warps=8,
            )
    else:
        ids, probs = _nucleus_reference(rows, mask_id, temperature, active)
    return ids.view(*shape, ids.shape[-1]), probs.view(*shape, ids.shape[-1])
