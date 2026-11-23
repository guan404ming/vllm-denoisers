from vllm.config.diffusion import DiffusionConfig
from vllm.model_executor.models.config import VerifyAndUpdateConfig

from denoisers.scheduler import select_scheduler


class SigmaModelConfig(VerifyAndUpdateConfig):
    @staticmethod
    def verify_and_update_model_config(model_config):
        if model_config.logits_processors:
            raise ValueError("Sigma does not support custom logits processors.")
        model_config.logits_processors = ["denoisers.models.sigma.params:SigmaSamplingParams"]
        model_config.hf_config.schedule_parameterization = "projected_bernstein"
        model_config.hf_config.canvas_length = model_config.hf_config.block_size

    @staticmethod
    def verify_and_update_config(vllm_config):
        model = vllm_config.model_config
        if model.quantization is not None:
            raise ValueError("Sigma needs unquantized checkpoint weights.")
        if not model.enforce_eager:
            raise ValueError("Sigma needs --enforce-eager.")
        parallel = vllm_config.parallel_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise ValueError("Sigma needs a single GPU per replica.")
        canvas = model.hf_config.block_size
        if vllm_config.diffusion_config is None:
            vllm_config.diffusion_config = DiffusionConfig(canvas_length=canvas)
        if vllm_config.diffusion_config.canvas_length != canvas:
            raise ValueError(f"Sigma needs canvas_length {canvas}.")
        scheduler = vllm_config.scheduler_config
        if scheduler.max_num_batched_tokens < scheduler.max_num_seqs * canvas:
            raise ValueError("Sigma needs max_num_batched_tokens >= max_num_seqs * canvas_length.")
        if scheduler.async_scheduling:
            raise ValueError("Sigma needs --no-async-scheduling.")
        scheduler.async_scheduling = False
        select_scheduler(scheduler)
