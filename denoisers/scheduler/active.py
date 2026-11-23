from collections import defaultdict, deque
from dataclasses import dataclass
from statistics import median

from denoisers.policies.hold import HoldPolicy

WINDOW = 20
MIN_EFFICIENCY = 0.97


class DecodeTimings:
    def __init__(self):
        self.samples = defaultdict(lambda: deque(maxlen=WINDOW))
        self.cost = {}
        self.limited_steps = defaultdict(int)
        self.pending = {}
        self.last_completion = 0.0
        self.last_size = 0

    def submitted(self, step, size, now):
        self.pending[id(step)] = (size, now)

    def completed(self, step, now):
        size, start = self.pending.pop(id(step), (0, now))
        if size and (start >= self.last_completion or size == self.last_size):
            duration = now - max(start, self.last_completion)
            if duration > 0:
                samples = self.samples[size]
                samples.append(duration)
                if len(samples) == WINDOW:
                    self.cost[size] = median(samples)
        self.last_completion = now
        self.last_size = size

    def target(self, size):
        full = self.cost.get(size)
        if full is None:
            return size
        target = min(
            (
                k
                for k, cost in self.cost.items()
                if k < size and cost < full and k * full >= MIN_EFFICIENCY * size * cost
            ),
            default=size,
        )
        if target < size:
            self.limited_steps[size] += 1
            if self.limited_steps[size] >= WINDOW:
                self.limited_steps[size] = 0
                return size
        return target


@dataclass(slots=True)
class DecodeProgress:
    output_tokens: int
    last_block: float
    last_scheduled: float


class DynamicActive:
    def __init__(self):
        self.timings = DecodeTimings()
        self.progress: dict[str, DecodeProgress] = {}

    def held(self, decode, can_hold: bool, policy: HoldPolicy, scores, now, wall_time):
        active = self.timings.target(len(decode)) if can_hold else len(decode)
        progress = {}
        candidates = []
        for request in decode:
            rid = request.request_id
            state = self.progress.get(rid)
            if state is None:
                age = max(0.0, wall_time - request.arrival_time)
                state = DecodeProgress(0, now - age, now)
            if request.num_output_tokens != state.output_tokens:
                state.output_tokens = request.num_output_tokens
                state.last_block = now
            progress[rid] = state
            block_limit = 1.5 if state.output_tokens else 3.75
            if (
                active < len(decode)
                and now - state.last_scheduled < 0.1
                and now - state.last_block < block_limit
            ):
                candidates.append((rid, request.arrival_time))
        self.progress = progress
        if active == len(decode):
            return set()
        protected = len(decode) - len(candidates)
        return policy.held(candidates, wall_time, max(0, active - protected), scores)

    def record_scheduled(self, scheduled, now):
        for rid in scheduled:
            if rid in self.progress:
                self.progress[rid].last_scheduled = now
