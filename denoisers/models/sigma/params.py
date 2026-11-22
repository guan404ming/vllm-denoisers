from vllm.v1.worker.gpu.sample.logits_processor import LogitsProcessor


class SigmaSamplingParams(LogitsProcessor):
    @classmethod
    def validate_params(cls, params):
        if (
            params.logprobs is not None
            or params.prompt_logprobs is not None
            or params.logprob_token_ids
            or params.prompt_logprob_token_ids is not None
        ):
            raise ValueError("Sigma returns token IDs and does not expose logprobs.")
        if (
            params.top_p != 1
            or params.top_k > 0
            or params.presence_penalty
            or params.frequency_penalty
            or params.repetition_penalty != 1
            or params.extra_args
        ):
            raise ValueError("Sigma uses the checkpoint's sampling recipe.")

    def apply(self, logits, ctx):
        return logits
