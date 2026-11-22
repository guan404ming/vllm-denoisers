import random

PRIOR = 0.0


def held(
    candidates: list[tuple[str, float]],
    now: float,
    policy: str,
    active: int,
    age_s: float,
    scores: dict[str, float],
) -> set[str]:
    if policy == "random":
        ranked = random.sample(candidates, len(candidates))
    else:
        rate = 1.0 / age_s if policy == "aging" else 0.0
        ranked = sorted(
            candidates,
            key=lambda c: scores.get(c[0], PRIOR) - rate * (now - c[1]),
        )
    return {rid for rid, _ in ranked[active:]}
