from denoisers.scheduler.hold.interface import PRIOR, HoldPolicy, registry


@registry.register("sjf")
class ShortestFirst(HoldPolicy):
    def rank(
        self, candidates: list[tuple[str, float]], now: float, scores: dict[str, float]
    ) -> list[str]:
        return [rid for rid, _ in sorted(candidates, key=lambda c: scores.get(c[0], PRIOR))]
