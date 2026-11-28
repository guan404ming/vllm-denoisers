from vllm.config import set_current_vllm_config
from vllm.config.diffusion import DiffusionConfig
from vllm.logger import init_logger
from vllm.model_executor.models.config import VerifyAndUpdateConfig
from vllm.v1.attention.backends.fa_utils import get_flash_attn_version
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from denoisers.models.common.config import DecodeConfig
from denoisers.policies import decode, feedback, kv
from denoisers.scheduler import select_scheduler
from denoisers.scheduler.scores import uses_probe

logger = init_logger("vllm.denoisers")

BACKENDS = (AttentionBackendEnum.TRITON_ATTN, AttentionBackendEnum.FLASH_ATTN)
GRAPH_SIZES = (
    "cudagraph_capture_sizes",
    "max_cudagraph_capture_size",
    "decode_cudagraph_capture_request_sizes",
    "max_decode_cudagraph_capture_requests",
)


def default_backend(vllm_config, policy) -> AttentionBackendEnum:
    with set_current_vllm_config(vllm_config):
        version = get_flash_attn_version(
            head_size=vllm_config.model_config.get_head_size(), supports_fa4_hd256=True
        )
    if version != 4 or policy.causal_commit or vllm_config.cache_config.sliding_window is not None:
        return AttentionBackendEnum.TRITON_ATTN
    return AttentionBackendEnum.FLASH_ATTN


def speculation_width(vllm_config, hf_config, policy, name: str) -> int:
    canvas = hf_config.canvas_length
    if not getattr(hf_config, "speculate", False):
        return canvas
    if (
        not issubclass(policy, decode.ScoreMovePolicy)
        or policy.causal_commit
        or not policy.aligned
        or policy.history_length
        or policy.init_canvas is not decode.DecodePolicy.init_canvas
        or getattr(hf_config, "feedback_policy", None) is not None
        or getattr(hf_config, "kv_policy", None) is not None
        or uses_probe(vllm_config.scheduler_config)
    ):
        raise ValueError(
            f"{name} speculate needs a block-aligned score and move decode_policy "
            "with a bidirectional commit and no kv, feedback, or probe policy."
        )
    attention = vllm_config.attention_config
    if attention.backend not in (None, AttentionBackendEnum.FLASH_ATTN) or (
        attention.flash_attn_version not in (None, 4)
    ):
        raise ValueError(f"{name} speculate needs --attention-backend FLASH_ATTN on FA4.")
    attention.backend = AttentionBackendEnum.FLASH_ATTN
    attention.flash_attn_version = 4
    return canvas * (4 if policy.causal_prefill and not policy.shifted else 2)


def verify_block_diffusion(vllm_config, hf_config, name: str) -> None:
    policy = decode.registry.resolve(hf_config.decode_policy)
    if getattr(hf_config, "max_post_steps", 1) < 1:
        raise ValueError("max_post_steps must be at least 1.")
    scheduler = vllm_config.scheduler_config
    if getattr(hf_config, "speculate", False) and scheduler.scheduler_cls is None:
        scheduler.scheduler_cls = "elastic"
    select_scheduler(scheduler, align_prefill=policy.aligned and scheduler.enable_chunked_prefill)
    width = speculation_width(vllm_config, hf_config, policy, name)
    if vllm_config.diffusion_config is None:
        vllm_config.diffusion_config = DiffusionConfig(canvas_length=width)
    if vllm_config.diffusion_config.canvas_length != width:
        raise ValueError(f"{name} needs canvas_length {width}.")
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
    if width > canvas and getattr(scheduler.get_scheduler_cls(), "elastic_widths", False):
        hf_config.speculate_widths = list(range(canvas, width + 1, canvas))
    fit = scheduler.max_num_batched_tokens // width
    if not fit:
        raise ValueError(f"{name} needs max_num_batched_tokens >= canvas_length {width}.")
    if fit < scheduler.max_num_seqs and not uses_probe(scheduler):
        logger.info("%s: max_num_seqs %d -> %d to fit the batch", name, scheduler.max_num_seqs, fit)
        scheduler.max_num_seqs = fit
    compilation = vllm_config.compilation_config
    if all(getattr(compilation, size) is None for size in GRAPH_SIZES):
        compilation.max_decode_cudagraph_capture_requests = scheduler.max_num_seqs
    if getattr(hf_config, "feedback_policy", None) is not None:
        feedback.registry.resolve(hf_config.feedback_policy)
    if getattr(hf_config, "kv_policy", None) is not None:
        kv.registry.resolve(hf_config.kv_policy)
        if scheduler.async_scheduling:
            raise ValueError(f"{name} kv_policy needs --no-async-scheduling.")
        scheduler.async_scheduling = False
    attention = vllm_config.attention_config
    if attention.backend is None:
        attention.backend = default_backend(vllm_config, policy)
    if attention.backend not in BACKENDS:
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
