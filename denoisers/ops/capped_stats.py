import torch

from denoisers.ops.stats import stable_jit

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None

BLOCK = 4096
CANDIDATES = 128
MIN_PROB = 1e-3


def softcap(logits: torch.Tensor, cap: float | None) -> torch.Tensor:
    logits = logits.float()
    if cap is None:
        return logits
    capped = torch.tanh(logits / cap) * cap
    return torch.where(logits == -torch.inf, logits, capped)


if triton is not None:

    @stable_jit
    def _capped_row_stats_kernel(
        logits_ptr,
        logits_stride,
        temps_ptr,
        argmax_ptr,
        sample_ptr,
        entropy_ptr,
        probs_ptr,
        probs_stride,
        ids_ptr,
        weights_ptr,
        blocks_ptr,
        seed,
        cap,
        min_prob,
        V,
        CL,
        NB,
        CAPPED: tl.constexpr,
        SPARSE: tl.constexpr,
        K: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        temp = tl.load(temps_ptr + row // CL).to(tl.float32)
        inv = 1.0 / tl.maximum(temp, 1e-10)
        base = logits_ptr + row.to(tl.int64) * logits_stride
        lanes = tl.arange(0, BLOCK)
        m = float("-inf")
        z = 0.0
        s = 0.0
        best = float("-inf")
        best_idx = 0
        for start in range(0, V, BLOCK):
            offs = start + lanes
            mask = offs < V
            x = tl.load(base + offs, mask=mask, other=float("-inf")).to(tl.float32)
            if CAPPED:
                t = 1.0 - 2.0 / (tl.exp(2.0 * x / cap) + 1.0)
                x = tl.where(x == float("-inf"), x, t * cap)
            block_max = tl.max(x, 0)
            new_m = tl.maximum(m, block_max)
            safe_m = tl.where(new_m > float("-inf"), new_m, 0.0)
            alpha = tl.where(m > float("-inf"), tl.exp((m - safe_m) * inv), 0.0)
            shift = tl.where(z > 0, (m - safe_m) * inv * z, 0.0)
            d = (x - safe_m) * inv
            e = tl.exp(d)
            block_sum = tl.sum(e, 0)
            z = z * alpha + block_sum
            s = (s + shift) * alpha + tl.sum(tl.where(e > 0, d * e, 0.0), 0)
            m = new_m
            block_idx = tl.argmax(x, 0)
            take = block_max > best
            best_idx = tl.where(take, start + block_idx, best_idx)
            best = tl.where(take, block_max, best)
            if SPARSE:
                stats = blocks_ptr + (row.to(tl.int64) * NB + start // BLOCK) * 4
                tl.store(stats, block_max)
                tl.store(stats + 1, safe_m)
                tl.store(stats + 2, block_sum)
        tl.store(argmax_ptr + row, best_idx)
        tl.store(entropy_ptr + row, tl.log(z) - s / z)
        target = tl.rand(seed, row) * z
        seen = 0.0
        sample_idx = best_idx
        found = (temp <= 0).to(tl.float32)
        out = probs_ptr + row.to(tl.int64) * probs_stride
        bag = row.to(tl.int64) * K
        count = 0
        if SPARSE:
            thresh = m + tl.log(min_prob * z) / inv
            for k in range(NB):
                stats = blocks_ptr + (row.to(tl.int64) * NB + k) * 4
                block_max = tl.load(stats)
                mass = tl.load(stats + 2) * tl.exp((tl.load(stats + 1) - m) * inv)
                take_s = (found == 0.0) & (seen + mass >= target) & (mass > 0)
                cand = block_max >= thresh
                if take_s | cand:
                    start = k * BLOCK
                    offs = start + lanes
                    mask = offs < V
                    x = tl.load(base + offs, mask=mask, other=float("-inf")).to(tl.float32)
                    if CAPPED:
                        t = 1.0 - 2.0 / (tl.exp(2.0 * x / cap) + 1.0)
                        x = tl.where(x == float("-inf"), x, t * cap)
                    e = tl.exp((x - m) * inv)
                    if take_s:
                        reach = seen + tl.cumsum(e, 0)
                        hit = tl.min(tl.where((reach >= target) & (e > 0), lanes, BLOCK), 0)
                        hit = tl.where(hit < BLOCK, hit, tl.argmax(e, 0))
                        sample_idx = start + hit
                        found = 1.0
                    if cand:
                        p = e / z
                        flags = ((p >= min_prob) & mask).to(tl.int32)
                        pos = count + tl.cumsum(flags, 0) - flags
                        keep = (flags > 0) & (pos < K)
                        tl.store(ids_ptr + bag + pos, offs, mask=keep)
                        tl.store(weights_ptr + bag + pos, p, mask=keep)
                        count += tl.sum(flags, 0)
                seen += mass
            pad = tl.arange(0, K)
            unused = pad >= count
            tl.store(ids_ptr + bag + pad, tl.zeros([K], dtype=tl.int64), mask=unused)
            tl.store(weights_ptr + bag + pad, tl.zeros([K], dtype=tl.float32), mask=unused)
        else:
            for start in range(0, V, BLOCK):
                offs = start + lanes
                mask = offs < V
                x = tl.load(base + offs, mask=mask, other=float("-inf")).to(tl.float32)
                if CAPPED:
                    t = 1.0 - 2.0 / (tl.exp(2.0 * x / cap) + 1.0)
                    x = tl.where(x == float("-inf"), x, t * cap)
                e = tl.exp((x - m) * inv)
                reach = seen + tl.cumsum(e, 0)
                hit = tl.min(tl.where((reach >= target) & (e > 0), lanes, BLOCK), 0)
                take_s = (hit < BLOCK) & (found == 0.0)
                sample_idx = tl.where(take_s, start + hit, sample_idx)
                found = tl.maximum(found, take_s.to(tl.float32))
                seen += tl.sum(e, 0)
                tl.store(out + offs, (e / z).to(probs_ptr.dtype.element_ty), mask=mask)
        tl.store(sample_ptr + row, sample_idx)


def select_candidates(probs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rows = probs.shape[0]
    flags = probs >= MIN_PROB
    pos = flags.cumsum(dim=1) - 1
    keep = flags & (pos < CANDIDATES)
    ids = torch.zeros(rows, CANDIDATES, dtype=torch.int64, device=probs.device)
    weights = torch.zeros(rows, CANDIDATES, dtype=torch.float32, device=probs.device)
    r, c = keep.nonzero(as_tuple=True)
    ids[r, pos[r, c]] = c
    weights[r, pos[r, c]] = probs[r, c].float()
    return ids, weights


def capped_row_stats_reference(logits, temps, block, cap, probs_dtype, sparse=False):
    temp = temps.repeat_interleave(block).float().unsqueeze(1)
    scaled = softcap(logits, cap) / temp.clamp(min=1e-10)
    noise = torch.rand_like(scaled).clamp(min=1e-20)
    noisy = scaled - torch.log(-torch.log(noise)) * (temp > 0).float()
    log_probs = scaled.log_softmax(dim=-1)
    probs = log_probs.exp()
    entropy = -torch.where(probs > 0, probs * log_probs, 0.0).sum(dim=-1)
    out = select_candidates(probs) if sparse else probs.to(probs_dtype)
    return scaled.argmax(dim=-1), noisy.argmax(dim=-1), entropy, out


def capped_row_stats(logits, temps, block, cap, probs_dtype, sparse=False):
    if not logits.is_cuda or triton is None:
        return capped_row_stats_reference(logits, temps, block, cap, probs_dtype, sparse)
    rows, vocab = logits.shape
    device = logits.device
    argmax = torch.empty(rows, dtype=torch.int64, device=device)
    sample = torch.empty(rows, dtype=torch.int64, device=device)
    entropy = torch.empty(rows, dtype=torch.float32, device=device)
    blocks_per_row = (vocab + BLOCK - 1) // BLOCK
    if sparse:
        probs = entropy
        ids = torch.empty(rows, CANDIDATES, dtype=torch.int64, device=device)
        weights = torch.empty(rows, CANDIDATES, dtype=torch.float32, device=device)
        blocks = torch.empty(rows, blocks_per_row, 4, dtype=torch.float32, device=device)
    else:
        probs = torch.empty(rows, vocab, dtype=probs_dtype, device=device)
        ids = weights = blocks = entropy
    if rows:
        seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        _capped_row_stats_kernel[(rows,)](
            logits,
            logits.stride(0),
            temps,
            argmax,
            sample,
            entropy,
            probs,
            probs.stride(0),
            ids,
            weights,
            blocks,
            seed,
            0.0 if cap is None else cap,
            MIN_PROB,
            vocab,
            block,
            blocks_per_row,
            CAPPED=cap is not None,
            SPARSE=sparse,
            K=CANDIDATES,
            BLOCK=BLOCK,
            num_warps=8,
        )
    return argmax, sample, entropy, (ids, weights) if sparse else probs
