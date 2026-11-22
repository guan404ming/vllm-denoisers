import torch
from torch.nn import functional as F

try:
    from vllm.distributed import tensor_model_parallel_all_reduce
except ImportError:
    tensor_model_parallel_all_reduce = None


class ExpectedEmbedding:
    def __init__(
        self,
        weight: torch.Tensor,
        vocab_start: int = 0,
        vocab_end: int | None = None,
        tp_size: int = 1,
        scale: torch.Tensor | float = 1.0,
    ):
        self.weight = weight
        self.vocab_start = vocab_start
        self.vocab_end = weight.shape[0] if vocab_end is None else vocab_end
        self.tp_size = tp_size
        self.scale = scale

    @classmethod
    def of(cls, embedding, tp_size: int = 1, scale: torch.Tensor | float = 1.0):
        shard = embedding.shard_indices
        return cls(
            embedding.weight,
            shard.org_vocab_start_index,
            shard.org_vocab_end_index,
            tp_size,
            scale,
        )

    @property
    def hidden_size(self) -> int:
        return self.weight.shape[1]

    def __call__(self, probs) -> torch.Tensor:
        size = self.vocab_end - self.vocab_start
        weight = self.weight[:size]
        if isinstance(probs, tuple):
            ids, weights = probs
            ids = ids - self.vocab_start
            local = (ids >= 0) & (ids < size)
            out = F.embedding_bag(
                ids.clamp(0, size - 1),
                weight,
                per_sample_weights=(weights * local).to(weight.dtype),
                mode="sum",
            )
        else:
            out = probs[:, self.vocab_start : self.vocab_end].to(weight.dtype) @ weight
        if self.tp_size > 1:
            out = tensor_model_parallel_all_reduce(out)
        return out * self.scale
