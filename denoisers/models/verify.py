from vllm.config.diffusion import DiffusionConfig
from vllm.model_executor.models.config import VerifyAndUpdateConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from denoisers.models.config import DecodeConfig
from denoisers.policies import decode, inputs, kv
from denoisers.scheduler.probe import uses_denoisers_scheduler

BACKENDS = (AttentionBackendEnum.TRITON_ATTN, AttentionBackendEnum.FLASH_ATTN)


def verify_block_diffusion(vllm_config, hf_config, name: str) -> None:
    policy = decode.registry.resolve(hf_config.decode_policy)
    if getattr(hf_config, "max_post_steps", 1) < 1:
        raise ValueError("max_post_steps must be at least 1.")
    if vllm_config.diffusion_config is None:
        vllm_config.diffusion_config = DiffusionConfig(canvas_length=hf_config.canvas_length)
    if vllm_config.diffusion_config.canvas_length != hf_config.canvas_length:
        raise ValueError(f"{name} needs canvas_length {hf_config.canvas_length}.")
    scheduler = vllm_config.scheduler_config
    if scheduler.max_num_batched_tokens < scheduler.max_num_seqs * hf_config.canvas_length:
        raise ValueError(f"{name} needs max_num_batched_tokens >= max_num_seqs * canvas_length.")
    if vllm_config.parallel_config.pipeline_parallel_size > 1:
        raise ValueError(f"{name} does not support pipeline parallelism.")
    if vllm_config.parallel_config.enable_batch_sharded_sampling:
        raise ValueError(f"{name} does not support batch-sharded sampling.")
    cache = vllm_config.cache_config
    canvas = hf_config.canvas_length
    aligned = policy.aligned
    if (
        aligned
        and cache.enable_prefix_caching
        and (cache.block_size % canvas or (cache.prefix_match_unit or 0) % canvas)
    ):
        raise ValueError(
            f"{name} prefix caching needs --block-size (and any --prefix-match-unit) "
            f"that is a multiple of canvas_length {canvas}."
        )
    if aligned and scheduler.enable_chunked_prefill and not uses_denoisers_scheduler(scheduler):
        raise ValueError(
            f"{name} needs --no-enable-chunked-prefill, or a denoisers scheduler-cls "
            "that aligns prefill chunks to the canvas."
        )
    if getattr(hf_config, "input_policy", None) is not None:
        inputs.registry.resolve(hf_config.input_policy)
    if getattr(hf_config, "kv_policy", None) is not None:
        kv.registry.resolve(hf_config.kv_policy)
        if scheduler.async_scheduling:
            raise ValueError(f"{name} kv_policy needs --no-async-scheduling.")
        scheduler.async_scheduling = False
    if vllm_config.attention_config.backend not in BACKENDS:
        raise ValueError(f"{name} needs --attention-backend TRITON_ATTN or FLASH_ATTN.")


class BlockDiffusionModelConfig(VerifyAndUpdateConfig):
    @classmethod
    def verify_and_update_config(cls, vllm_config) -> None:
        hf_config = vllm_config.model_config.hf_config
        if not isinstance(hf_config, DecodeConfig):
            raise TypeError(
                f"{type(hf_config).__name__} is not a denoisers config; drop --trust-remote-code."
            )
        hf_config.verify(vllm_config)
        verify_block_diffusion(vllm_config, hf_config, hf_config.model_type)
