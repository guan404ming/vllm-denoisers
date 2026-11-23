from dataclasses import dataclass

from denoisers.policies.hold import HoldPolicy

ACTIVE_LIMITS = {9: 8, 10: 8, 17: 16, 19: 16}


@dataclass(slots=True)
class DecodeProgress:
    output_tokens: int
    last_block: float
    last_scheduled: float


class DynamicActive:
    def __init__(self):
        self.progress: dict[str, DecodeProgress] = {}

    def held(self, decode, can_hold: bool, policy: HoldPolicy, scores, now, wall_time):
        active = ACTIVE_LIMITS.get(len(decode)) if can_hold else None
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
                active is not None
                and now - state.last_scheduled < 0.1
                and now - state.last_block < block_limit
            ):
                candidates.append((rid, request.arrival_time))
        self.progress = progress
        if active is None:
            return set()
        protected = len(decode) - len(candidates)
        return policy.held(candidates, wall_time, max(0, active - protected), scores)

    def record_scheduled(self, scheduled, now):
        for rid in scheduled:
            if rid in self.progress:
                self.progress[rid].last_scheduled = now
