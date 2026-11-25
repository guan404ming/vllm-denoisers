from abc import ABC, abstractmethod

from denoisers.utils.registry import Registry

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
        reserve: int = 0,
    ) -> set[str]:
        held = self.rank(candidates, now, scores)[active - reserve :]
        return set(held) - set(sorted(held, key=dict(candidates).get)[:reserve])


registry = Registry("hold policy", HoldPolicy)
