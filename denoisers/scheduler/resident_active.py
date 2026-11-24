import os
import time
from collections import deque

from vllm.logger import init_logger
from vllm.v1.core.sched.diffusion_scheduler import diffusion_canvas_width

from denoisers.policies.hold import PRIOR, registry
from denoisers.scheduler import probe
from denoisers.scheduler.aligned import AlignedAsyncScheduler, AlignedScheduler
from denoisers.scheduler.rounds import DecodeTimings

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
        self.hold = registry.resolve(HOLD).from_env()
        self.scores = probe.ScoreTable(
            probe.table_path(vllm_config), vllm_config.scheduler_config.max_num_seqs
        )

        self.timings = DecodeTimings()
        self.progress = {}
        self.parts = ()
        self.groups = deque()

    def _next_group(self, decode):
        progress = {rid: r.num_output_tokens for rid, r in decode.items()}
        changed = progress != self.progress
        if changed:
            if progress.keys() != self.progress.keys():
                self.groups.clear()
            self.parts = ()
            self.progress = progress
        if self.groups:
            return self.groups.popleft()
        if changed or len(decode) < 2 or len(decode) not in self.timings.cost:
            return None
        if not self.parts:
            self.parts = self.timings.partition(len(decode))
        if len(self.parts) > 1:
            candidates = [(rid, r.arrival_time) for rid, r in decode.items()]
            ranked = self.hold.rank(candidates, time.time(), self.scores.read(list(decode)))
            start = 0
            for size in self.parts:
                self.groups.append(set(ranked[start : start + size]))
                start += size
            return self.groups.popleft()
        return None

    def schedule(self, throttle_prefills=False):
        start = time.monotonic_ns()
        decode = {
            r.request_id: r for r in self.running if r.num_computed_tokens >= r.num_prompt_tokens
        }
        uniform = (
            not self.waiting
            and not self.kv_holding_waiting
            and len(decode) == len(self.running)
            and all(
                diffusion_canvas_width(r, self.num_spec_tokens) == self.num_spec_tokens
                and r.next_decode_eligible_step <= self.current_step + 1
                for r in decode.values()
            )
        )
        group = self._next_group(decode if uniform else {})
        held = decode.keys() - group if group is not None else set()
        for rid in held:
            r = decode[rid]
            r.next_decode_eligible_step = max(r.next_decode_eligible_step, self.current_step + 2)
        if self.current_step % 50 == 0 and decode:
            scores = self.scores.read(list(decode))
            ranked = [scores.get(rid, PRIOR) for rid in decode]
            logger.info(
                "denoisers: hold=%s step=%d decode=%d held=%d scored=%d score=%.2f/%.2f",
                HOLD,
                self.current_step,
                len(decode),
                len(held),
                len(scores),
                min(ranked),
                max(ranked),
            )
        result = super().schedule(throttle_prefills)
        scheduled = result.num_scheduled_tokens
        if group is not None and scheduled.keys() != group:
            self.groups.clear()
            self.progress.clear()
        if (
            uniform
            and scheduled
            and all(
                rid in decode and width == self.num_spec_tokens for rid, width in scheduled.items()
            )
        ):
            self.timings.submitted(result, len(scheduled), start)
        return result

    def update_from_output(self, scheduler_output, model_runner_output):
        self.timings.completed(scheduler_output, time.monotonic_ns())
        return super().update_from_output(scheduler_output, model_runner_output)


class DenoisersScheduler(ResidentActiveScheduler):
    pass


class DenoisersAsyncScheduler(ResidentActiveScheduler, AlignedAsyncScheduler):
    pass
