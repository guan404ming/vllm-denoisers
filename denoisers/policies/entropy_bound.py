from dataclasses import replace

import torch
from torch.nn import functional as F

from denoisers.canvas.transition import CanvasRows
from denoisers.policies.capped_stats import capped_row_stats, softcap

try:
    from vllm.distributed import (
        get_tensor_model_parallel_world_size,
        tensor_model_parallel_all_reduce,
    )
except ImportError:
    get_tensor_model_parallel_world_size = tensor_model_parallel_all_reduce = None

COEF = (1.1628, 0.0352, -3.6599, 1.2418, 0.2033, 6.0314)


def remaining_steps(entropy: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
    mean, median, low, top, per_step, bias = COEF
    predicted = (
        mean * entropy.mean(dim=-1)
        + median * entropy.median(dim=-1).values
        + low * (entropy < 0.1).float().mean(dim=-1)
        + top * entropy.max(dim=-1).values
        + per_step * step.float()
        + bias
    )
    return predicted.clamp(min=0.5)


class EntropyBound:
    aligned = False
    causal_commit = True
    commit_logits = False

    def __init__(
        self,
        vocab_size: int,
        max_steps: int,
        t_min: float,
        t_max: float,
        entropy_bound: float,
        confidence_threshold: float,
        embed_weight: torch.Tensor,
        normalizer: torch.Tensor,
        vocab_start: int = 0,
        vocab_end: int | None = None,
        tp_size: int = 1,
        sparse: bool = False,
        softcap: float | None = None,
    ):
        self.vocab_size = vocab_size
        self.max_steps = max_steps
        self.t_min = t_min
        self.t_max = t_max
        self.entropy_bound = entropy_bound
        self.confidence_threshold = confidence_threshold
        self.embed_weight = embed_weight
        self.normalizer = normalizer
        self.vocab_start = vocab_start
        self.vocab_end = vocab_size if vocab_end is None else vocab_end
        self.tp_size = tp_size
        self.sparse = sparse
        self.softcap = softcap
        self.hidden_size = embed_weight.shape[1]
        self.slot_steps = None
        self.constrained: set[int] = set()

    @classmethod
    def from_config(cls, config, model) -> "EntropyBound":
        embed = model.model.embed_tokens
        shard = embed.shard_indices
        return cls(
            vocab_size=embed.org_vocab_size,
            max_steps=config.max_denoising_steps,
            t_min=config.t_min,
            t_max=config.t_max,
            entropy_bound=config.entropy_bound,
            confidence_threshold=config.confidence_threshold,
            embed_weight=embed.weight,
            normalizer=model.model.normalizer,
            vocab_start=shard.org_vocab_start_index,
            vocab_end=shard.org_vocab_end_index,
            tp_size=get_tensor_model_parallel_world_size(),
            sparse=config.sparse_self_conditioning,
            softcap=model.final_logit_softcapping,
        )

    def bind(self, state) -> None:
        rows, block = state.canvas.shape
        device = state.canvas.device
        self.slot_steps = torch.full((rows,), self.max_steps, dtype=torch.int64, device=device)
        self.seed = torch.zeros((rows, block), dtype=torch.int64, device=device)
        self.seeded = torch.zeros(rows, dtype=torch.bool, device=device)
        self.pinned = torch.zeros((rows, block), dtype=torch.bool, device=device)
        self.read_only = torch.zeros(rows, dtype=torch.bool, device=device)
        self.allowed = torch.ones((rows, self.vocab_size), dtype=torch.bool, device=device)

    def add_request(self, slot: int, sampling_params) -> None:
        extra = getattr(sampling_params, "extra_args", None) or {}
        device = self.seed.device
        cap = extra.get("diffusion_max_steps")
        steps = self.max_steps if cap is None else max(1, min(int(cap), self.max_steps))
        self.slot_steps[slot] = steps
        seed = extra.get("diffusion_seed_canvas")
        self.seeded[slot] = seed is not None
        self.pinned[slot] = False
        if seed is not None:
            self.seed[slot, : len(seed)] = torch.tensor(seed, dtype=torch.int64, device=device)
            pins = extra.get("diffusion_pinned") or []
            if pins:
                self.pinned[slot, torch.tensor(pins, dtype=torch.int64, device=device)] = True
        self.read_only[slot] = bool(extra.get("diffusion_read_only"))
        self.constrained.discard(slot)
        self.allowed[slot] = True
        if extra.get("diffusion_constrained"):
            ids = list(getattr(sampling_params, "logprob_token_ids", None) or [])
            if not ids:
                raise ValueError("diffusion_constrained needs logprob_token_ids.")
            self.allowed[slot] = False
            self.allowed[slot, torch.tensor(ids, dtype=torch.int64, device=device)] = True
            self.constrained.add(slot)

    def init_canvas(self, canvas: torch.Tensor, slots=None, first: bool = False) -> torch.Tensor:
        noise = torch.randint(
            0, self.vocab_size, canvas.shape, dtype=canvas.dtype, device=canvas.device
        )
        if first and self.slot_steps is not None:
            return torch.where(self.seeded[slots].unsqueeze(1), self.seed[slots], noise)
        return noise

    def temperature(self, step: torch.Tensor) -> torch.Tensor:
        remaining = (self.max_steps - step.float()).clamp(min=1.0)
        return self.t_min + (self.t_max - self.t_min) * remaining / self.max_steps

    def stats(self, logits: torch.Tensor, temps: torch.Tensor, block: int):
        return capped_row_stats(
            logits, temps, block, self.softcap, self.embed_weight.dtype, self.sparse
        )

    def logprob_logits(self, logits: torch.Tensor) -> torch.Tensor:
        return softcap(logits, self.softcap)

    def soft_embeds(self, probs) -> torch.Tensor:
        size = self.vocab_end - self.vocab_start
        if isinstance(probs, tuple):
            ids, weights = probs
            ids = ids - self.vocab_start
            local = (ids >= 0) & (ids < size)
            soft = F.embedding_bag(
                ids.clamp(0, size - 1),
                self.embed_weight[:size],
                per_sample_weights=(weights * local).to(self.embed_weight.dtype),
                mode="sum",
            )
        else:
            soft = probs[:, self.vocab_start : self.vocab_end] @ self.embed_weight[:size]
        if self.tp_size > 1:
            soft = tensor_model_parallel_all_reduce(soft)
        return soft * self.normalizer

    def __call__(
        self, logits: torch.Tensor, rows: CanvasRows
    ) -> tuple[CanvasRows, torch.Tensor]:
        n, block, vocab = logits.shape
        device = logits.device
        bound = self.slot_steps is not None and rows.slots is not None
        if bound and self.constrained:
            logits = torch.where(self.allowed[rows.slots].unsqueeze(1), logits, -torch.inf)
        argmax, sample, entropy, probs = self.stats(
            logits.view(-1, vocab), self.temperature(rows.step), block
        )
        argmax, sample, entropy = (t.view(n, block) for t in (argmax, sample, entropy))
        valid = torch.arange(block, device=device) < rows.width.unsqueeze(1)
        entropy = torch.where(valid, entropy, 0.0)
        confident = entropy.sum(dim=1) / rows.width.clamp(min=1) < self.confidence_threshold
        order = entropy.argsort(dim=1)
        ordered = entropy.gather(1, order)
        accepted = ordered.cumsum(dim=1) - ordered.cummax(dim=1).values <= self.entropy_bound
        accept = torch.zeros_like(accepted).scatter_(1, order, accepted)
        canvas = torch.where(accept, sample, self.init_canvas(rows.canvas, rows.slots))
        history = rows.history.clone()
        history[torch.arange(n, device=device), rows.history_len % history.shape[1]] = argmax
        history_len = rows.history_len + 1
        stable = (history == history[:, :1]).all(dim=2).all(dim=1)
        stable &= history_len >= history.shape[1]
        max_steps = self.slot_steps[rows.slots] if bound else self.max_steps
        converged = (stable & confident) | (rows.step + 1 >= max_steps)
        canvas = torch.where(converged.unsqueeze(1), argmax, canvas)
        keep = valid & ~converged.unsqueeze(1)
        emit = None
        if bound:
            pinned = self.pinned[rows.slots]
            canvas = torch.where(pinned, self.seed[rows.slots], canvas)
            keep &= ~pinned
            emit = converged & self.read_only[rows.slots]
        soft = self.soft_embeds(probs).view(n, block, -1) * keep.unsqueeze(2)
        after = replace(
            rows,
            canvas=canvas,
            history=history,
            history_len=history_len,
            soft=soft.float(),
            entropy=entropy,
            emit=emit,
        )
        return after, converged

    def probe_score(self, entropy: torch.Tensor, step: torch.Tensor) -> torch.Tensor:
        return remaining_steps(entropy, step)
