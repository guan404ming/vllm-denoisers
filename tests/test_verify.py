from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from vllm.config.scheduler import SchedulerConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum

from denoisers.models.verify import verify_block_diffusion
from denoisers.scheduler.probe import uses_probe
from denoisers.scheduler.resident_active import DenoisersScheduler

BLOCK = 8


def make_config():
    scheduler = SimpleNamespace(
        max_num_batched_tokens=BLOCK,
        max_num_seqs=1,
        enable_chunked_prefill=False,
        async_scheduling=None,
    )
    config = SimpleNamespace(
        diffusion_config=SimpleNamespace(canvas_length=BLOCK),
        scheduler_config=scheduler,
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1, enable_batch_sharded_sampling=False
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        attention_config=SimpleNamespace(backend=AttentionBackendEnum.FLASH_ATTN),
    )
    hf_config = SimpleNamespace(
        decode_policy="low_confidence", canvas_length=BLOCK, kv_policy="masked"
    )
    return config, hf_config


def test_kv_policy_turns_off_default_async_scheduling():
    config, hf_config = make_config()
    verify_block_diffusion(config, hf_config, "test")
    assert config.scheduler_config.async_scheduling is False


class ExternalScheduler(DenoisersScheduler):
    pass


class AlignedScheduler(ExternalScheduler):
    uses_denoiser_scores = False


@pytest.mark.parametrize("scheduler_cls", [ExternalScheduler, f"{__name__}.ExternalScheduler"])
def test_chunked_prefill_uses_inherited_scheduler_capability(scheduler_cls):
    config, hf_config = make_config()
    scheduler = config.scheduler_config
    scheduler.enable_chunked_prefill = True
    scheduler.scheduler_cls = scheduler_cls
    scheduler.get_scheduler_cls = lambda: SchedulerConfig.get_scheduler_cls(scheduler)
    verify_block_diffusion(config, hf_config, "test")
    assert uses_probe(scheduler)
    scheduler.scheduler_cls = AlignedScheduler
    assert not uses_probe(scheduler)
    scheduler.scheduler_cls = object
    with pytest.raises(ValueError, match="chunked-prefill"):
        verify_block_diffusion(config, hf_config, "test")
