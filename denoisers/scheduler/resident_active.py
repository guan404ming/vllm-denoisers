import os
import time

from vllm.logger import init_logger
from vllm.v1.core.sched.diffusion_scheduler import (
    DiffusionAsyncScheduler,
    DiffusionScheduler,
    diffusion_canvas_width,
)

from denoisers.canvas.batch import aligned_prefill_chunk
from denoisers.scheduler import probe
from denoisers.scheduler.policy import PRIOR, held
from denoisers.scheduler.sources import Oracle, Regressor, Trace, prompt_key, random_scores

logger = init_logger("vllm.denoisers")

POLICY = os.environ.get("DENOISERS_POLICY", "sjf")
ACTIVE = int(os.environ.get("DENOISERS_ACTIVE", "16"))
AGE_S = float(os.environ.get("DENOISERS_AGE_S", "0"))
PROBE = os.environ.get("DENOISERS_PROBE", "entropy")
TRACE = os.environ.get("DENOISERS_TRACE")
SOURCE = os.environ.get("DENOISERS_SOURCE")
ALIGN = os.environ.get("DENOISERS_ALIGN", "1") != "0"

if POLICY == "aging" and AGE_S <= 0:
    raise ValueError("DENOISERS_POLICY=aging needs DENOISERS_AGE_S > 0.")


class _Denoisers:
    def __init__(self, vllm_config, *args, **kwargs):
        super().__init__(vllm_config, *args, **kwargs)
        self.need_mamba_block_aligned_split = ALIGN
        self.scores = probe.ScoreTable(
            probe.table_path(vllm_config), vllm_config.scheduler_config.max_num_seqs
        )
        self.trace = Trace(TRACE) if TRACE else None
        self.oracle = Oracle(SOURCE) if PROBE == "oracle" else None
        self.regressor = Regressor(SOURCE) if PROBE == "regressor" else None
        self.elapsed: dict[str, int] = {}
        self.emitted: dict[str, int] = {}

    def _mamba_block_aligned_split(self, request, num_new_tokens, *computed) -> int:
        start = request.num_computed_tokens + sum(computed)
        return aligned_prefill_chunk(
            start, num_new_tokens, request.num_prompt_tokens, self.num_spec_tokens
        )

    def _track(self, decode) -> None:
        live = {r.request_id for r in decode}
        self.elapsed = {k: v for k, v in self.elapsed.items() if k in live}
        self.emitted = {k: v for k, v in self.emitted.items() if k in live}
        for r in decode:
            rid = r.request_id
            out = r.num_output_tokens
            if rid in self.emitted and out > self.emitted[rid]:
                canvas = diffusion_canvas_width(r, self.num_spec_tokens)
                if self.trace is not None:
                    self.trace.commit(
                        prompt_key(r.prompt_token_ids),
                        (out - 1) // canvas,
                        self.elapsed.get(rid, 0),
                        out,
                        r.num_prompt_tokens,
                    )
                self.elapsed[rid] = 0
            self.emitted[rid] = out
            self.elapsed.setdefault(rid, 0)

    def _scores(self, decode) -> dict[str, float]:
        if PROBE == "entropy":
            return self.scores.read([r.request_id for r in decode])
        if PROBE == "random":
            return random_scores([r.request_id for r in decode], 1.0, 20.0)
        scores = {}
        for r in decode:
            canvas = diffusion_canvas_width(r, self.num_spec_tokens)
            elapsed = self.elapsed.get(r.request_id, 0)
            if self.oracle is not None:
                block = r.num_output_tokens // canvas
                scores[r.request_id] = self.oracle.remaining(
                    prompt_key(r.prompt_token_ids), block, elapsed
                )
            else:
                scores[r.request_id] = self.regressor.remaining(
                    r.num_prompt_tokens, r.num_output_tokens, canvas, elapsed
                )
        return scores

    def schedule(self, throttle_prefills=False):
        decode = [
            r for r in self.running if r.num_computed_tokens >= r.num_prompt_tokens
        ]
        self._track(decode)
        scores = self._scores(decode)
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
            else:
                self.elapsed[r.request_id] += 1
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
