import math
import os
import time
from collections import Counter

from vllm.logger import init_logger
from vllm.v1.core.sched.diffusion_scheduler import diffusion_canvas_width
from vllm.v1.request import RequestStatus

from denoisers.policies.hold import registry
from denoisers.scheduler import probe
from denoisers.scheduler.aligned import AlignedAsyncScheduler, AlignedScheduler
from denoisers.scheduler.remaining import RemainingWork

logger = init_logger("vllm.denoisers")

HOLD = os.environ.get("DENOISERS_HOLD", "fair")
ACTIVE = int(os.environ.get("DENOISERS_ACTIVE", "32"))
LEARNED = (RequestStatus.FINISHED_STOPPED, RequestStatus.FINISHED_LENGTH_CAPPED)


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
        self.hold = registry.resolve(HOLD).from_env()
        self.scores = probe.ScoreTable(
            probe.table_path(vllm_config), vllm_config.scheduler_config.max_num_seqs
        )

        self.work = RemainingWork()
        self.steps = Counter()

    def schedule(self, throttle_prefills=False):
        decode = {
            r.request_id: r for r in self.running if r.num_computed_tokens >= r.num_prompt_tokens
        }
        probes = self.scores.read(list(decode))
        blocks = {
            rid: r.num_output_tokens // diffusion_canvas_width(r, self.num_spec_tokens)
            for rid, r in decode.items()
        }
        scores = {
            rid: self.work.estimate(rid, blocks[rid], self.steps[rid], probes.get(rid))
            for rid in decode
        }
        held = self.hold.held(
            [(rid, r.arrival_time) for rid, r in decode.items()], time.time(), ACTIVE, scores
        )
        for rid in held:
            r = decode[rid]
            r.next_decode_eligible_step = max(r.next_decode_eligible_step, self.current_step + 2)
        if self.current_step % 50 == 0 and decode:
            ranked = [score for score in scores.values() if score > -math.inf]
            logger.info(
                "denoisers: hold=%s step=%d decode=%d held=%d scored=%d score=%.2f/%.2f",
                HOLD,
                self.current_step,
                len(decode),
                len(held),
                len(probes),
                min(ranked, default=0.0),
                max(ranked, default=0.0),
            )
        result = super().schedule(throttle_prefills)
        for rid in result.num_scheduled_tokens:
            if rid in decode:
                self.steps[rid] += 1
                if scores[rid] == -math.inf:
                    self.work.probing(rid, blocks[rid])
        return result

    def _free_request(self, request, delay_free_blocks=False):
        steps = self.steps.pop(request.request_id, 0)
        self.work.finished(request.request_id, steps, request.status in LEARNED)
        return super()._free_request(request, delay_free_blocks)


class DenoisersScheduler(ResidentActiveScheduler):
    pass


class DenoisersAsyncScheduler(ResidentActiveScheduler, AlignedAsyncScheduler):
    pass
