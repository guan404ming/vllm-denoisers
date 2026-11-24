from denoisers.policies.hold.interface import PRIOR, HoldPolicy, registry


@registry.register("fair")
class FairShare(HoldPolicy):
    def __init__(self):
        self.clock = 0.0
        self.joined = {}

    def rank(
        self, candidates: list[tuple[str, float]], now: float, scores: dict[str, float]
    ) -> list[str]:
        def key(candidate):
            rid = candidate[0]
            return scores.get(rid, PRIOR) + self.joined.get(rid, self.clock)

        return [rid for rid, _ in sorted(candidates, key=key)]

    def held(
        self,
        candidates: list[tuple[str, float]],
        now: float,
        active: int,
        scores: dict[str, float],
    ) -> set[str]:
        self.joined = {rid: self.joined.get(rid, self.clock) for rid, _ in candidates}
        if candidates:
            self.clock += min(active, len(candidates)) / len(candidates)
        return super().held(candidates, now, active, scores)
