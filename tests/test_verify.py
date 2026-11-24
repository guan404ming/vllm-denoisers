from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from vllm.config.scheduler import SchedulerConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.core.sched.diffusion_scheduler import DiffusionAsyncScheduler, DiffusionScheduler

from denoisers.models.verify import verify_block_diffusion
from denoisers.scheduler.aligned import AlignedAsyncScheduler, AlignedScheduler
from denoisers.scheduler.probe import uses_probe
from denoisers.scheduler.resident_active import (
    DenoisersAsyncScheduler,
    DenoisersScheduler,
    ResidentActiveScheduler,
)

BLOCK = 8


def make_config():
    scheduler = SchedulerConfig(
        max_model_len=BLOCK,
        is_encoder_decoder=False,
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


@pytest.mark.parametrize("scheduler_cls", [ExternalScheduler, f"{__name__}.ExternalScheduler"])
def test_chunked_prefill_uses_inherited_scheduler_capability(scheduler_cls):
    config, hf_config = make_config()
    scheduler = config.scheduler_config
    scheduler.enable_chunked_prefill = True
    scheduler.scheduler_cls = scheduler_cls
    verify_block_diffusion(config, hf_config, "test")
    assert scheduler.scheduler_cls == scheduler_cls
    assert uses_probe(scheduler)
    scheduler.scheduler_cls = AlignedScheduler
    assert not uses_probe(scheduler)
    scheduler.scheduler_cls = object
    with pytest.raises(ValueError, match="chunked-prefill"):
        verify_block_diffusion(config, hf_config, "test")


@pytest.mark.parametrize(
    ("async_mode", "expected", "base"),
    [
        (True, AlignedAsyncScheduler, DiffusionAsyncScheduler),
        (False, AlignedScheduler, DiffusionScheduler),
    ],
)
def test_default_aligned_scheduler_uses_final_async_mode(monkeypatch, async_mode, expected, base):
    config, hf_config = make_config()
    hf_config.kv_policy = None
    scheduler = config.scheduler_config
    scheduler.enable_chunked_prefill = True
    verify_block_diffusion(config, hf_config, "test")
    assert scheduler.async_scheduling is None
    assert not uses_probe(scheduler)

    scheduler.async_scheduling = async_mode
    monkeypatch.setattr(DiffusionScheduler, "__init__", lambda self, vllm_config: None)
    instance = scheduler.get_scheduler_cls()(vllm_config=config)
    assert type(instance) is expected
    assert instance.schedule.__func__ is base.schedule
    instance.num_spec_tokens = BLOCK
    instance.num_prefill_lookahead = 0
    request = SimpleNamespace(num_prompt_tokens=21, num_tokens=21)
    assert instance._reserve_prefill_lookahead(request, 8, 10) == 8
    assert instance._reserve_prefill_lookahead(request, 16, 5) == 5


@pytest.mark.parametrize(
    ("policy", "chunked"), [("nemotron_diffusion", True), ("low_confidence", False)]
)
def test_other_modes_keep_native_scheduler(policy, chunked):
    config, hf_config = make_config()
    hf_config.decode_policy = policy
    config.scheduler_config.enable_chunked_prefill = chunked
    verify_block_diffusion(config, hf_config, "test")
    assert config.scheduler_config.scheduler_cls is None


@pytest.mark.parametrize(
    ("name", "async_mode", "expected"),
    [
        ("resact", False, DenoisersScheduler),
        ("resact", True, DenoisersAsyncScheduler),
        ("aligned", False, AlignedScheduler),
        ("aligned", True, AlignedAsyncScheduler),
    ],
)
def test_scheduler_short_names_follow_final_mode(monkeypatch, name, async_mode, expected):
    config, hf_config = make_config()
    hf_config.kv_policy = None
    config.instance_id = "short-name-test"
    config.parallel_config.data_parallel_rank = 0
    scheduler = config.scheduler_config
    scheduler.scheduler_cls = name
    verify_block_diffusion(config, hf_config, "test")
    assert uses_probe(scheduler) == (name == "resact")
    scheduler.async_scheduling = async_mode
    monkeypatch.setattr(DiffusionScheduler, "__init__", lambda self, vllm_config: None)
    assert type(scheduler.get_scheduler_cls()(vllm_config=config)) is expected


@pytest.mark.parametrize("async_mode", [False, True])
def test_resact_splits_a_round_and_interrupts_for_admission(monkeypatch, async_mode):
    config, _ = make_config()
    config.instance_id = "adaptive-resact"
    config.parallel_config.data_parallel_rank = 0
    config.scheduler_config.async_scheduling = async_mode
    monkeypatch.setattr(DiffusionScheduler, "__init__", lambda self, config: None)
    scheduler = ResidentActiveScheduler(config)
    scheduler.waiting = []
    scheduler.kv_holding_waiting = []
    scheduler.current_step = 0
    scheduler.num_spec_tokens = BLOCK
    scheduler.running = [
        SimpleNamespace(
            request_id=rid,
            arrival_time=0,
            num_output_tokens=0,
            num_computed_tokens=1,
            num_prompt_tokens=1,
            next_decode_eligible_step=0,
            sampling_params=None,
        )
        for rid in ("a", "b", "c")
    ]
    scheduler.timings.cost.update({1: 2, 2: 3, 3: 6})
    monkeypatch.setattr(scheduler.scores, "read", lambda ids: {"a": 2, "b": 1, "c": 0})

    def dispatch(self, throttle_prefills):
        self.current_step += 1
        batch = {
            r.request_id: BLOCK
            for r in self.running
            if r.next_decode_eligible_step <= self.current_step
        }
        return SimpleNamespace(num_scheduled_tokens=batch)

    monkeypatch.setattr(AlignedScheduler, "schedule", dispatch)

    def step():
        return set(scheduler.schedule().num_scheduled_tokens)

    assert step() == {"a", "b", "c"}
    assert step() == {"b", "c"}
    assert step() == {"a"}
    assert step() == {"b", "c"}
    scheduler.waiting.append("new")
    assert step() == {"a", "b", "c"}
    scheduler.waiting.clear()
    assert step() == {"a", "b", "c"}
    assert step() == {"b", "c"}
    scheduler.running.pop(0)
    assert step() == {"b", "c"}
