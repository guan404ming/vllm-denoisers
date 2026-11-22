from types import SimpleNamespace

from vllm.v1.attention.backends.registry import AttentionBackendEnum

from denoisers.models.verify import verify_block_diffusion

BLOCK = 8


def test_kv_policy_turns_off_default_async_scheduling():
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
    verify_block_diffusion(config, hf_config, "test")
    assert scheduler.async_scheduling is False
