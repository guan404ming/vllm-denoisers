import collections

from denoisers.scheduler import widths
from denoisers.scheduler.aligned import AlignedAsyncScheduler, AlignedScheduler
from denoisers.scheduler.widths import (
    best_width,
    bucket,
    cost_at,
    fit_costs,
    gain_bound,
    median_floor,
)

WINDOW = 64


class ElasticScheduler(AlignedScheduler):
    elastic_widths = True

    def __new__(cls, vllm_config, *args, **kwargs):
        if cls is ElasticScheduler and vllm_config.scheduler_config.async_scheduling:
            cls = ElasticAsyncScheduler
        return super().__new__(cls, vllm_config, *args, **kwargs)

    def __init__(self, vllm_config, *args, **kwargs):
        super().__init__(vllm_config, *args, **kwargs)
        config = vllm_config.model_config.hf_config
        if not getattr(config, "speculate_costs", None):
            raise ValueError(
                "Elastic speculation needs CUDA graphs in one process to probe costs. "
                "Pass --scheduler-cls aligned for a fixed width."
            )
        self.block = config.canvas_length
        self.costs = fit_costs(config.speculate_costs)
        self.gains = tuple(range(1, vllm_config.diffusion_config.canvas_length // self.block + 1))
        self.excess: dict[tuple[int, int], collections.deque] = {}
        widths.gains = None
        widths.periods.clear()
        self.dynamic_sd_lookup = self

    def cost(self, size: int, rows: int) -> float:
        seen = self.excess.get((rows // (size * self.block), bucket(rows)), ())
        return cost_at(self.costs, rows) * (median_floor(seen) if len(seen) >= 3 else 1.0)

    def __getitem__(self, size: int) -> int:
        return self.block * best_width(size, self.block, self.cost, self.gains)

    def schedule(self, throttle_prefills: bool = False):
        while widths.periods:
            size, rows, seconds = widths.periods.popleft()
            key = (rows // (size * self.block), bucket(rows))
            self.excess.setdefault(key, collections.deque(maxlen=WINDOW)).append(
                1000 * seconds / cost_at(self.costs, rows)
            )
        if widths.gains is not None:
            advance, forwards = widths.gains
            self.gains = tuple(
                gain_bound(float(advance[w]), float(forwards[w]), w)
                for w in range(1, len(self.gains) + 1)
            )
        return super().schedule(throttle_prefills)


class ElasticAsyncScheduler(ElasticScheduler, AlignedAsyncScheduler):
    pass
