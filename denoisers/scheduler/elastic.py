import collections

from vllm.logger import init_logger

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

logger = init_logger("vllm.denoisers")

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
            logger.warning(
                "Elastic speculation needs CUDA graphs in a single process to probe costs. "
                "Using the widest canvas at every step."
            )
            return
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
        while widths.periods:
            count, rows, seconds = widths.periods.popleft()
            key = (rows // (count * self.block), bucket(rows))
            self.excess.setdefault(key, collections.deque(maxlen=WINDOW)).append(
                1000 * seconds / cost_at(self.costs, rows)
            )
        if widths.gains is not None:
            advance, forwards = widths.gains
            self.gains = tuple(
                gain_bound(float(advance[w]), float(forwards[w]), w)
                for w in range(1, len(self.gains) + 1)
            )
        return self.block * best_width(size, self.block, self.cost, self.gains)


class ElasticAsyncScheduler(ElasticScheduler, AlignedAsyncScheduler):
    pass
