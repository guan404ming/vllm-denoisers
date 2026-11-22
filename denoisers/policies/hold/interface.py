from abc import ABC, abstractmethod

from denoisers.registry import Registry

PRIOR = 0.0


class HoldPolicy(ABC):
    @classmethod
    def from_env(cls) -> "HoldPolicy":
        return cls()

    @abstractmethod
    def rank(
        self, candidates: list[tuple[str, float]], now: float, scores: dict[str, float]
    ) -> list[str]:
        raise NotImplementedError

    def held(
        self,
        candidates: list[tuple[str, float]],
        now: float,
        active: int,
        scores: dict[str, float],
    ) -> set[str]:
        return set(self.rank(candidates, now, scores)[active:])


registry = Registry("hold policy", HoldPolicy)
