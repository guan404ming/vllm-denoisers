import os
import time

from vllm.logger import init_logger
from vllm.v1.core.sched.diffusion_scheduler import DiffusionAsyncScheduler, DiffusionScheduler

from denoisers.canvas.batch import aligned_prefill_chunk
from denoisers.scheduler import probe
from denoisers.scheduler.policy import PRIOR, held

logger = init_logger("vllm.denoisers")

POLICY = os.environ.get("DENOISERS_POLICY", "sjf")
ACTIVE = int(os.environ.get("DENOISERS_ACTIVE", "16"))
AGE_S = float(os.environ.get("DENOISERS_AGE_S", "0"))

if POLICY == "aging" and AGE_S <= 0:
    raise ValueError("DENOISERS_POLICY=aging needs DENOISERS_AGE_S > 0.")


class _Denoisers:
    def __init__(self, vllm_config, *args, **kwargs):
        super().__init__(vllm_config, *args, **kwargs)
        self.need_mamba_block_aligned_split = True
        self.scores = probe.ScoreTable(
            probe.table_path(vllm_config), vllm_config.scheduler_config.max_num_seqs
        )

    def _mamba_block_aligned_split(self, request, num_new_tokens, *computed) -> int:
        start = request.num_computed_tokens + sum(computed)
        return aligned_prefill_chunk(
            start, num_new_tokens, request.num_prompt_tokens, self.num_spec_tokens
        )

    def schedule(self, throttle_prefills=False):
        decode = [
            r for r in self.running if r.num_computed_tokens >= r.num_prompt_tokens
        ]
        scores = self.scores.read([r.request_id for r in decode])
        hold = held(
            [(r.request_id, r.arrival_time) for r in decode],
            time.time(),
            POLICY,
            ACTIVE,
            AGE_S,
            scores,
        )
        for r in decode:
            if r.request_id in hold:
                r.next_decode_eligible_step = max(
                    r.next_decode_eligible_step, self.current_step + 2
                )
        if self.current_step % 50 == 0 and decode:
            ranked = [scores.get(r.request_id, PRIOR) for r in decode]
            logger.info(
                "denoisers: policy=%s step=%d decode=%d held=%d scored=%d score=%.2f/%.2f",
                POLICY,
                self.current_step,
                len(decode),
                len(hold),
                len(scores),
                min(ranked),
                max(ranked),
            )
        return super().schedule(throttle_prefills)


class DenoisersScheduler(_Denoisers, DiffusionScheduler):
    pass


class DenoisersAsyncScheduler(_Denoisers, DiffusionAsyncScheduler):
    pass
