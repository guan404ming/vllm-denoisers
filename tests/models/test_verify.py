from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("vllm")

from vllm.config.scheduler import SchedulerConfig
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.core.sched.diffusion_scheduler import DiffusionAsyncScheduler, DiffusionScheduler

from denoisers.canvas.state import CanvasState
from denoisers.models.common import verify
from denoisers.models.common.verify import GRAPH_SIZES, verify_block_diffusion
from denoisers.runtime.model_state import BlockDiffusionModelState, ranged
from denoisers.scheduler.aligned import AlignedAsyncScheduler, AlignedScheduler
from denoisers.scheduler.resident_active import (
    DenoisersAsyncScheduler,
    DenoisersScheduler,
)
from denoisers.scheduler.scores import uses_probe

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
        cache_config=SimpleNamespace(enable_prefix_caching=False, sliding_window=None),
        model_config=SimpleNamespace(get_head_size=lambda: 64),
        attention_config=SimpleNamespace(backend=AttentionBackendEnum.FLASH_ATTN),
        compilation_config=SimpleNamespace(**dict.fromkeys(GRAPH_SIZES)),
    )
    hf_config = SimpleNamespace(
        decode_policy="low_confidence", canvas_length=BLOCK, kv_policy="masked"
    )
    return config, hf_config


def test_kv_policy_turns_off_default_async_scheduling():
    config, hf_config = make_config()
    verify_block_diffusion(config, hf_config, "test")
    assert config.scheduler_config.async_scheduling is False


def test_sequences_fit_the_batch_token_budget():
    config, hf_config = make_config()
    scheduler = config.scheduler_config
    scheduler.max_num_batched_tokens, scheduler.max_num_seqs = 5 * BLOCK, 1024
    verify_block_diffusion(config, hf_config, "test")
    assert scheduler.max_num_seqs == 5
    scheduler.max_num_seqs = 3
    verify_block_diffusion(config, hf_config, "test")
    assert scheduler.max_num_seqs == 3
    scheduler.max_num_batched_tokens = BLOCK - 1
    with pytest.raises(ValueError, match="canvas_length"):
        verify_block_diffusion(config, hf_config, "test")


def test_decode_graphs_cover_every_sequence():
    config, hf_config = make_config()
    scheduler, compilation = config.scheduler_config, config.compilation_config
    scheduler.max_num_batched_tokens, scheduler.max_num_seqs = 5 * BLOCK, 1024
    verify_block_diffusion(config, hf_config, "test")
    assert compilation.max_decode_cudagraph_capture_requests == 5
    compilation.max_decode_cudagraph_capture_requests = None
    compilation.max_cudagraph_capture_size = 64
    verify_block_diffusion(config, hf_config, "test")
    assert compilation.max_decode_cudagraph_capture_requests is None


def test_resact_keeps_its_resident_count():
    config, hf_config = make_config()
    scheduler = config.scheduler_config
    scheduler.scheduler_cls = "resact"
    scheduler.max_num_batched_tokens, scheduler.max_num_seqs = 5 * BLOCK, 1024
    verify_block_diffusion(config, hf_config, "test")
    assert scheduler.max_num_seqs == 1024


@pytest.mark.parametrize(
    ("policy", "version", "window", "expected"),
    [
        ("low_confidence", 4, None, AttentionBackendEnum.FLASH_ATTN),
        ("low_confidence", 3, None, AttentionBackendEnum.TRITON_ATTN),
        ("low_confidence", None, None, AttentionBackendEnum.TRITON_ATTN),
        ("low_confidence", 4, 1024, AttentionBackendEnum.TRITON_ATTN),
        ("dream", 4, None, AttentionBackendEnum.FLASH_ATTN),
        ("dream", 4, 1024, AttentionBackendEnum.TRITON_ATTN),
        ("entropy_bound", 4, None, AttentionBackendEnum.TRITON_ATTN),
    ],
)
def test_attention_backend_defaults_to_flash_when_it_fits(
    monkeypatch, policy, version, window, expected
):
    config, hf_config = make_config()
    hf_config.decode_policy = policy
    config.attention_config.backend = None
    config.cache_config.sliding_window = window
    monkeypatch.setattr(verify, "get_flash_attn_version", lambda **_: version)
    verify_block_diffusion(config, hf_config, "test")
    assert config.attention_config.backend is expected


def test_attention_backend_keeps_a_supported_choice():
    config, hf_config = make_config()
    config.attention_config.backend = AttentionBackendEnum.TRITON_ATTN
    verify_block_diffusion(config, hf_config, "test")
    assert config.attention_config.backend is AttentionBackendEnum.TRITON_ATTN
    config.attention_config.backend = AttentionBackendEnum.FLASHINFER
    with pytest.raises(ValueError, match="TRITON_ATTN or FLASH_ATTN"):
        verify_block_diffusion(config, hf_config, "test")


def test_flash_ranges_need_fa4_global_layers_and_a_bidirectional_commit():
    def model(version, window):
        decoder = SimpleNamespace(
            attn_type=AttentionType.DECODER, vllm_flash_attn_version=version, sliding_window=window
        )
        encoder = SimpleNamespace(
            attn_type=AttentionType.ENCODER_ONLY, vllm_flash_attn_version=2, sliding_window=(3, 3)
        )
        layers = [SimpleNamespace(impl=decoder), SimpleNamespace(impl=encoder), object()]
        return SimpleNamespace(modules=lambda: layers)

    def policy(aligned=True, causal_prefill=True, causal_commit=False):
        return SimpleNamespace(
            aligned=aligned, causal_prefill=causal_prefill, causal_commit=causal_commit
        )

    flash, triton = AttentionBackendEnum.FLASH_ATTN, AttentionBackendEnum.TRITON_ATTN
    assert ranged(flash, policy(), model(4, (-1, -1)))
    assert not ranged(triton, policy(), model(4, (-1, -1)))
    assert not ranged(flash, policy(aligned=False, causal_commit=True), model(4, (1023, 0)))
    for bad_policy, bad_model in (
        (policy(), model(3, (-1, -1))),
        (policy(), model(4, (1023, 0))),
        (policy(causal_commit=True), model(4, (-1, -1))),
    ):
        with pytest.raises(ValueError, match="TRITON_ATTN"):
            ranged(flash, bad_policy, bad_model)


def test_block_ranges_cover_prefilling_rows_only():
    runtime = object.__new__(BlockDiffusionModelState)
    runtime.aligned = runtime.causal_prefill = True
    runtime.state = SimpleNamespace(block_length=4)
    batch = SimpleNamespace(
        num_reqs=3,
        cu_num_logits_np=np.array([0, 0, 0, 4]),
        query_start_loc_np=np.array([0, 6, 7, 11]),
        num_computed_prefill_tokens_np=np.array([0, 5, 6]),
        is_prefilling_np=np.array([True, False, False]),
    )
    assert runtime._doc_ranges(batch) == {0: [(0, 3), (4, 5)]}


def test_dummy_positions_stay_below_max_model_len():
    runtime = object.__new__(BlockDiffusionModelState)
    runtime.device = "cpu"
    runtime.model_config = SimpleNamespace(max_model_len=8)
    runtime.num_route_blocks = 64
    runtime.state = SimpleNamespace(block_length=4)
    runtime._positions = torch.zeros(20, dtype=torch.int64)
    runtime._block_ids = torch.zeros(20, dtype=torch.int64)
    runtime._inputs_embeds = None
    runtime.model = None
    positions = runtime.prepare_dummy_inputs(1, 20)["positions"]
    assert positions.tolist() == list(range(8)) * 2 + list(range(4))


def test_removed_request_leaves_a_neutral_slot_for_dummy_runs():
    runtime = object.__new__(BlockDiffusionModelState)
    runtime.state = CanvasState(3, 4, 9, "cpu")
    runtime.req_states = SimpleNamespace(req_id_to_index={"a": 1})
    runtime.state.begin(1, torch.tensor([5, 6]), 30)
    runtime.remove_request("a")
    runtime.remove_request("missing")
    assert runtime.state.shift.tolist() == [0, 0, 0]
    assert runtime.state.start.tolist() == [0, 0, 0]


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


@pytest.mark.parametrize(
    ("policy", "count"),
    [("low_confidence", 4), ("llada", 2), ("dream", 2), ("fast_dllm", 2)],
)
def test_speculate_widens_the_canvas_on_fa4(policy, count):
    config, hf_config = make_config()
    config.diffusion_config = None
    config.attention_config = SimpleNamespace(backend=None, flash_attn_version=None)
    config.scheduler_config.max_num_batched_tokens = count * BLOCK
    hf_config.decode_policy, hf_config.kv_policy, hf_config.speculate = policy, None, True
    verify_block_diffusion(config, hf_config, "test")
    assert config.diffusion_config.canvas_length == count * BLOCK
    assert config.attention_config.backend == AttentionBackendEnum.FLASH_ATTN
    assert config.attention_config.flash_attn_version == 4


@pytest.mark.parametrize(
    "change",
    [
        {"kv_policy": "masked"},
        {"decode_policy": "nemotron_diffusion"},
        {"decode_policy": "indel"},
        {"backend": AttentionBackendEnum.TRITON_ATTN},
    ],
)
def test_speculate_rejects_what_the_tree_cannot_verify(change):
    config, hf_config = make_config()
    config.diffusion_config = None
    config.attention_config = SimpleNamespace(
        backend=change.pop("backend", None), flash_attn_version=None
    )
    hf_config.kv_policy, hf_config.speculate = None, True
    vars(hf_config).update(change)
    with pytest.raises(ValueError, match="speculate"):
        verify_block_diffusion(config, hf_config, "test")
