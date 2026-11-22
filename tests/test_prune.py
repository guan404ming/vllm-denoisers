from types import SimpleNamespace

import torch
from denoisers.canvas.prune import active_rows, inverse_index
from denoisers.canvas.state import CanvasState
from denoisers.models.verify import verify_block_diffusion
from vllm.v1.attention.backends.registry import AttentionBackendEnum

MASK = 9
BLOCK = 8


def test_active_rows_keeps_masked_first_and_full_rows_identity():
    masked = torch.tensor(
        [
            [False, True, False, True, True, False, False, False],
            [True, True, True, True, True, True, True, True],
            [False, False, True, False, False, False, False, False],
        ]
    )
    width = torch.full((3,), BLOCK)
    full = torch.tensor([False, False, True])
    active, length = active_rows(masked, width, full, shifted=False)
    assert length.tolist() == [3, BLOCK, BLOCK]
    assert active[0, :3].tolist() == [1, 3, 4]
    assert sorted(active[0].tolist()) == list(range(BLOCK))
    assert active[1].tolist() == list(range(BLOCK))
    assert active[2].tolist() == list(range(BLOCK))


def test_active_rows_shifted_includes_left_neighbour():
    masked = torch.tensor([[False, False, True, False, False, True, False, False]])
    active, length = active_rows(masked, torch.tensor([BLOCK]), torch.tensor([False]), True)
    assert length.tolist() == [4]
    assert active[0, :4].tolist() == [1, 2, 4, 5]


def test_active_rows_respects_truncated_width():
    masked = torch.tensor([[True, True, True, True, True, False, False, False]])
    active, length = active_rows(masked, torch.tensor([5]), torch.tensor([False]), False)
    assert length.tolist() == [BLOCK]


def test_active_rows_without_masked_positions_is_full():
    masked = torch.zeros(1, BLOCK, dtype=torch.bool)
    active, length = active_rows(masked, torch.tensor([BLOCK]), torch.tensor([False]), False)
    assert length.tolist() == [BLOCK]
    assert active[0].tolist() == list(range(BLOCK))


def test_inverse_index_maps_block_offsets_to_hidden_rows():
    active = torch.tensor([[1, 3, 4, 0, 2, 5, 6, 7], list(range(BLOCK))])
    widths = torch.tensor([3, BLOCK])
    starts = torch.tensor([0, 3])
    inv = inverse_index(active, widths, starts)
    assert inv[0].tolist() == [-1, 0, -1, 1, 2, -1, -1, -1]
    assert inv[1].tolist() == [3, 4, 5, 6, 7, 8, 9, 10]


def test_state_reset_restores_full_active_rows():
    state = CanvasState(2, BLOCK, MASK, torch.device("cpu"))
    state.active[1] = torch.tensor([3, 0, 1, 2, 4, 5, 6, 7])
    state.active_len[1] = 1
    state.reset(1)
    assert state.active[1].tolist() == list(range(BLOCK))
    assert state.active_len[1].item() == BLOCK


def test_prune_turns_off_default_async_scheduling():
    scheduler = SimpleNamespace(
        max_num_batched_tokens=BLOCK, max_num_seqs=1, enable_chunked_prefill=False, async_scheduling=None
    )
    config = SimpleNamespace(
        diffusion_config=SimpleNamespace(canvas_length=BLOCK),
        scheduler_config=scheduler,
        parallel_config=SimpleNamespace(pipeline_parallel_size=1, enable_batch_sharded_sampling=False),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        attention_config=SimpleNamespace(backend=AttentionBackendEnum.FLASH_ATTN),
    )
    hf_config = SimpleNamespace(decode_policy="low_confidence", canvas_length=BLOCK, prune_committed=True)
    verify_block_diffusion(config, hf_config, "test")
    assert scheduler.async_scheduling is False
