import os
import time

from vllm.logger import init_logger

from denoisers.policies.hold import PRIOR, registry
from denoisers.scheduler import probe
from denoisers.scheduler.active import DynamicActive
from denoisers.scheduler.aligned import AlignedAsyncScheduler, AlignedScheduler

logger = init_logger("vllm.denoisers")

HOLD = os.environ.get("DENOISERS_HOLD", "sjf")


class ResidentActiveScheduler(AlignedScheduler):
    uses_denoiser_scores = True

    def __new__(cls, vllm_config, *args, **kwargs):
        if cls is ResidentActiveScheduler:
            cls = (
                DenoisersAsyncScheduler
                if vllm_config.scheduler_config.async_scheduling
                else DenoisersScheduler
            )
        return super().__new__(cls, vllm_config, *args, **kwargs)

    def __init__(self, vllm_config, *args, **kwargs):
        super().__init__(vllm_config, *args, **kwargs)
        active = os.environ.get("DENOISERS_ACTIVE", "auto")
        self.active = None if active == "auto" else int(active)
        if self.active is not None and self.active < 1:
            raise ValueError("DENOISERS_ACTIVE must be auto or a positive integer.")
        self.dynamic = DynamicActive() if self.active is None else None
        self.hold = registry.resolve(HOLD).from_env()
        self.scores = probe.ScoreTable(
            probe.table_path(vllm_config), vllm_config.scheduler_config.max_num_seqs
        )

    def schedule(self, throttle_prefills=False):
        decode = [r for r in self.running if r.num_computed_tokens >= r.num_prompt_tokens]
        scores = self.scores.read([r.request_id for r in decode])
        wall_time = time.time()
        if self.dynamic is None:
            hold = self.hold.held(
                [(r.request_id, r.arrival_time) for r in decode], wall_time, self.active, scores
            )
        else:
            now = time.monotonic()
            can_hold = (
                not self.waiting
                and not self.kv_holding_waiting
                and len(decode) == len(self.running)
            )
            hold = self.dynamic.held(decode, can_hold, self.hold, scores, now, wall_time)
        for r in decode:
            if r.request_id in hold:
                r.next_decode_eligible_step = max(
                    r.next_decode_eligible_step, self.current_step + 2
                )
        if self.current_step % 50 == 0 and decode:
            ranked = [scores.get(r.request_id, PRIOR) for r in decode]
            logger.info(
                "denoisers: hold=%s step=%d decode=%d held=%d scored=%d score=%.2f/%.2f",
                HOLD,
                self.current_step,
                len(decode),
                len(hold),
                len(scores),
                min(ranked),
                max(ranked),
            )
        result = super().schedule(throttle_prefills)
        if self.dynamic is not None:
            self.dynamic.record_scheduled(result.num_scheduled_tokens, now)
        return result


class DenoisersScheduler(ResidentActiveScheduler):
    pass


class DenoisersAsyncScheduler(ResidentActiveScheduler, AlignedAsyncScheduler):
    pass
