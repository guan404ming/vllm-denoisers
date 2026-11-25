from vllm.v1.core.sched.diffusion_scheduler import DiffusionAsyncScheduler, DiffusionScheduler

from denoisers.scheduler import aligned_prefill_chunk


class AlignedScheduler(DiffusionScheduler):
    aligns_denoiser_prefill = True

    def __new__(cls, vllm_config, *args, **kwargs):
        if cls is AlignedScheduler and vllm_config.scheduler_config.async_scheduling:
            cls = AlignedAsyncScheduler
        return super().__new__(cls)

    def _reserve_prefill_lookahead(self, request, num_computed_tokens, num_new_tokens):
        count = super()._reserve_prefill_lookahead(request, num_computed_tokens, num_new_tokens)
        return aligned_prefill_chunk(
            num_computed_tokens, count, request.num_prompt_tokens, self.num_spec_tokens
        )


class AlignedAsyncScheduler(AlignedScheduler, DiffusionAsyncScheduler):
    pass
