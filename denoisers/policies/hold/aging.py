import os

from denoisers.policies.hold.interface import PRIOR, HoldPolicy, registry


@registry.register("aging")
class Aging(HoldPolicy):
    def __init__(self, age_s: float):
        if age_s <= 0:
            raise ValueError("The aging hold policy needs DENOISERS_AGE_S > 0.")
        self.rate = 1.0 / age_s

    @classmethod
    def from_env(cls) -> "Aging":
        return cls(float(os.environ.get("DENOISERS_AGE_S", "0")))

    def rank(
        self, candidates: list[tuple[str, float]], now: float, scores: dict[str, float]
    ) -> list[str]:
        ranked = sorted(
            candidates, key=lambda c: scores.get(c[0], PRIOR) - self.rate * (now - c[1])
        )
        return [rid for rid, _ in ranked]
