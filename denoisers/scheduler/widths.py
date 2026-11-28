import collections
import math
import statistics

CONFIDENCE = math.log(20)

gains = None
periods: collections.deque = collections.deque(maxlen=1024)


def bucket(rows: float) -> int:
    return round(2 * math.log2(rows))


def gain_bound(advance: float, forwards: float, width: int) -> float:
    if width == 1:
        return 1.0
    if forwards <= 0:
        return float(width)
    slack = (width - 1) * math.sqrt(CONFIDENCE / (2 * forwards))
    return min(float(width), advance / forwards + slack)


def median_floor(samples) -> float:
    rank = 0.5 - math.sqrt(CONFIDENCE / (2 * len(samples)))
    return sorted(samples)[int(max(rank, 0.0) * len(samples))]


def cost_at(costs: list[tuple[float, float]], rows: float) -> float:
    if rows <= costs[0][0]:
        return costs[0][1]
    for (low, a), (high, b) in zip(costs, costs[1:]):
        if rows <= high:
            return a + (b - a) * (rows - low) / (high - low)
    (low, a), (high, b) = costs[-2:]
    return b + (b - a) * (rows - high) / (high - low)


def fit_costs(samples: list[tuple[float, float]]) -> list[tuple[float, float]]:
    groups: dict[int, list[tuple[float, float]]] = {}
    for rows, cost in samples:
        groups.setdefault(bucket(rows), []).append((rows, cost))
    grid, floor = [], 0.0
    for key in sorted(groups):
        members = groups[key]
        floor = max(floor, statistics.median(cost for _, cost in members))
        grid.append((statistics.mean(rows for rows, _ in members), floor))
    return grid


def best_width(size: int, block: int, cost, gains: list[float]) -> int:
    return max(
        range(1, len(gains) + 1),
        key=lambda w: gains[w - 1] / cost(max(size, 1), max(size, 1) * w * block),
    )
