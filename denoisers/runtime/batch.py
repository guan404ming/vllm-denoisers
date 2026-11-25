import numpy as np
import torch

FREE = (0, 1 << 30)


def finished_prefills(input_batch) -> np.ndarray:
    computed = input_batch.num_computed_prefill_tokens_np
    return np.flatnonzero(
        input_batch.is_prefilling_np
        & (computed + input_batch.num_scheduled_tokens >= input_batch.prefill_len_np)
    )


def query_lens(input_batch) -> torch.Tensor:
    loc = input_batch.query_start_loc[: input_batch.num_reqs + 1]
    return loc[1:] - loc[:-1]


def per_token(values: torch.Tensor, input_batch) -> torch.Tensor:
    return torch.repeat_interleave(
        values, query_lens(input_batch), output_size=input_batch.num_tokens
    )


def prefill_block_ranges(seq_len: int, block_length: int) -> list[tuple[int, int]]:
    return [
        (start, min(start + block_length, seq_len) - 1)
        for start in range(0, seq_len, block_length)
        if min(start + block_length, seq_len) - 1 > start
    ]


def free_ranges(ranges: dict, prefilling: np.ndarray) -> dict:
    return ranges | {int(row): [FREE] for row in np.flatnonzero(~prefilling)}


def aligned_prefill_chunk(start: int, num_new: int, prompt_len: int, block: int) -> int:
    end = start + num_new
    if end >= prompt_len:
        return num_new
    return max(end - end % block - start, 0)
