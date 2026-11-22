import torch


class BlockRouter:
    def __init__(
        self, gate, block_length: int, expert_capacity: int, routed_scaling_factor: float
    ):
        self.gate = gate
        self.block_length = block_length
        self.expert_capacity = expert_capacity
        self.routed_scaling_factor = routed_scaling_factor
        self.block_ids: torch.Tensor | None = None
        self.num_blocks = 0

    def set_blocks(self, block_ids: torch.Tensor, num_blocks: int) -> None:
        self.block_ids = block_ids
        self.num_blocks = num_blocks

    def _blocks(self, num_tokens: int) -> tuple[torch.Tensor, int]:
        if self.block_ids is None or self.block_ids.numel() < num_tokens:
            raise RuntimeError("block ids do not cover this forward's tokens.")
        return self.block_ids[:num_tokens], self.num_blocks

    def __call__(
        self,
        hidden_states: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.sigmoid(gating_output.float())
        routing = scores + self.gate.expert_bias
        num_tokens, num_experts = routing.shape
        ids, num_blocks = self._blocks(num_tokens)
        block_scores = routing.new_full((num_blocks, num_experts), -torch.inf)
        block_scores.scatter_reduce_(
            0, ids.unsqueeze(1).expand(num_tokens, num_experts), routing, "amax"
        )
        capacity = block_scores.topk(self.expert_capacity, dim=1).indices
        allowed = torch.zeros_like(block_scores, dtype=torch.bool)
        allowed.scatter_(1, capacity, True)
        topk_ids = routing.masked_fill(~allowed[ids], -torch.inf).topk(topk, dim=1).indices
        weights = scores.gather(1, topk_ids)
        if topk > 1:
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-20)
        return weights * self.routed_scaling_factor, topk_ids
