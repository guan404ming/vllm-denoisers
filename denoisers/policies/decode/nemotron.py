import torch

from denoisers.ops.sampling import sample
from denoisers.policies.decode.interface import ScoreMovePolicy, registry


@registry.register("nemotron_diffusion")
class NemotronDiffusion(ScoreMovePolicy):
    aligned = False
    causal_commit = True
    temperature = 0.0

    def __init__(self, model, mask_id: int, threshold: float | None):
        self.model = model
        self.mask_id = mask_id
        self.threshold = threshold

    @classmethod
    def from_config(cls, config, model):
        return cls(model, config.mask_token_id, config.confidence_threshold)

    def init_canvas(self, canvas, slots=None, first=False):
        if first:
            logits = self.model.compute_logits(self.model.prompt_hidden[slots])
            canvas[:, 0] = logits.argmax(-1)
        return canvas

    def next_seed(self, logits):
        return logits[:, -1].argmax(-1)

    def score(self, logits, noise=None, active=None):
        tokens, confidence = sample(logits, self.mask_id, self.temperature, active, noise)
        return tokens, confidence.to(logits.dtype)

    def move(self, canvas, step, shift, width, tokens, score):
        masked = canvas == self.mask_id
        chosen = torch.zeros_like(masked) if self.threshold is None else score >= self.threshold
        best = score.masked_fill(~masked, -torch.inf).argmax(-1, keepdim=True)
        chosen.scatter_(1, best, True)
        after = torch.where(masked & chosen, tokens, canvas)
        return after, ~(after == self.mask_id).any(-1)
