import random

from denoisers.scheduler.hold.interface import HoldPolicy, registry


@registry.register("random")
class RandomOrder(HoldPolicy):
    def rank(
        self, candidates: list[tuple[str, float]], now: float, scores: dict[str, float]
    ) -> list[str]:
        return [rid for rid, _ in random.sample(candidates, len(candidates))]
