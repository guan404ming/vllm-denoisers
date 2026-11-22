from typing import Any

import numpy as np
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.model_states.interface import ModelState

from denoisers.canvas.batch import per_token, prefill_block_ranges
from denoisers.canvas.sampler import BlockDiffusionSampler
from denoisers.canvas.state import CanvasState
from denoisers.scheduler import probe


def start_probe(vllm_config) -> None:
    if get_tensor_model_parallel_rank() == 0:
        probe.start(vllm_config)


class BlockDiffusionModelState(ModelState):
    num_new_sampled_tokens_per_step: int = 0

    def __init__(self, vllm_config, model, encoder_cache, device) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        start_probe(vllm_config)
        hf_config = self.model_config.hf_config
        self.policy = model.build_policy()
        self.aligned = getattr(self.policy, "aligned", True)
        self.causal_commit = getattr(self.policy, "causal_commit", False)
        self.causal_prefill = getattr(self.policy, "causal_prefill", True)
        self.prune = getattr(hf_config, "prune_committed", False)
        self.prune_refresh = getattr(hf_config, "prune_refresh", 0)
        hidden_size = getattr(self.policy, "hidden_size", None)
        if self.prune and hidden_size is not None:
            raise ValueError("prune_committed does not support self-conditioning policies.")
        self.state = CanvasState(
            self.max_num_reqs,
            hf_config.canvas_length,
            hf_config.mask_token_id,
            device,
            history_length=getattr(hf_config, "max_steps_per_block", 64),
            hidden_size=hidden_size,
        )
        self.max_blocks = -(-self.max_model_len // hf_config.canvas_length)
        self.num_route_blocks = self.max_num_reqs * self.max_blocks + 1
        self._causal_buf = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=device)
        self._seq_lens = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=device)
        self._positions = torch.zeros(self.max_num_tokens, dtype=torch.int64, device=device)
        self._block_ids = torch.zeros(self.max_num_tokens, dtype=torch.int64, device=device)
        self._inputs_embeds = (
            None
            if hidden_size is None
            else torch.zeros(
                self.max_num_tokens, hidden_size, dtype=self.model_config.dtype, device=device
            )
        )
        self._mm_embeds = None

    def get_supported_generation_tasks(self):
        return ("generate",)

    def custom_sampler(self, sampler: Any) -> tuple[Any, Any] | None:
        return BlockDiffusionSampler(
            sampler,
            self.state,
            self.policy,
            self.max_model_len,
            self.model.compute_logits,
            self.model_config.get_vocab_size(),
            prune_committed=self.prune,
            prune_refresh=self.prune_refresh,
        ), None

    def compute_sample_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def dummy_sampler_run(self, sampler: Any, hidden_states: torch.Tensor) -> bool:
        sampler.profile_run(hidden_states)
        return True

    def add_request(self, req_index: int, new_req_data: Any) -> None:
        self.state.reset(req_index)

    def _rows(self, input_batch) -> torch.Tensor:
        return per_token(
            torch.arange(input_batch.num_reqs, device=self.device), input_batch
        )

    def _token_shift(self, input_batch) -> torch.Tensor:
        slots = input_batch.idx_mapping[: input_batch.num_reqs]
        return per_token(self.state.shift[slots], input_batch)

    def _num_logits(self, input_batch) -> torch.Tensor:
        cu = input_batch.cu_num_logits[: input_batch.num_reqs + 1]
        return cu[1:] - cu[:-1]

    def _pruned(self, input_batch) -> torch.Tensor:
        slots = input_batch.idx_mapping[: input_batch.num_reqs]
        return (self.state.active_len[slots] < self.state.block_length) & (
            self._num_logits(input_batch) > 0
        )

    def _active_offset(self, input_batch) -> torch.Tensor:
        num_tokens = input_batch.num_tokens
        block = self.state.block_length
        rows = self._rows(input_batch)
        slots = input_batch.idx_mapping[: input_batch.num_reqs]
        local = torch.arange(num_tokens, device=self.device) - input_batch.query_start_loc[rows]
        active = self.state.active[slots][rows, local.clamp(max=block - 1)]
        return torch.where(self._pruned(input_batch)[rows], active - local, 0)

    def _set_blocks(self, ids: torch.Tensor) -> None:
        for router in getattr(self.model, "block_routers", ()):
            router.set_blocks(ids, self.num_route_blocks)

    def _fill_positions(self, input_batch) -> torch.Tensor:
        num_tokens = input_batch.num_tokens
        positions = self._positions[: input_batch.num_tokens_after_padding]
        positions.copy_(input_batch.positions[: positions.shape[0]])
        positions[:num_tokens] -= self._token_shift(input_batch)
        if self.prune:
            positions[:num_tokens] += self._active_offset(input_batch)
        return positions

    def _condition(self, input_batch) -> torch.Tensor:
        num_tokens = input_batch.num_tokens
        embeds = self._inputs_embeds[: input_batch.num_tokens_after_padding]
        input_ids = input_batch.input_ids[:num_tokens]
        if self._mm_embeds is None:
            embeds[:num_tokens].copy_(self.model.embed_input_ids(input_ids))
        else:
            mm_embeds, is_mm_embed = self._mm_embeds
            self._mm_embeds = None
            embeds[:num_tokens].copy_(
                self.model.embed_input_ids(
                    input_ids, multimodal_embeddings=mm_embeds, is_multimodal=is_mm_embed
                )
            )
        if input_batch.num_draft_tokens == 0:
            return embeds
        num_reqs = input_batch.num_reqs
        num_logits = np.diff(input_batch.cu_num_logits_np[: num_reqs + 1])
        rows = np.where(num_logits > 0)[0]
        slots_np = input_batch.idx_mapping_np[rows]
        starts = input_batch.query_start_loc_np[rows]
        widths = num_logits[rows]
        block = self.state.block_length
        if (widths == block).all():
            index = async_tensor_h2d(
                (starts[:, None] + np.arange(block)).reshape(-1), device=self.device
            )
            slots = async_tensor_h2d(slots_np.astype(np.int64), device=self.device)
            soft = self.state.soft[slots].view(-1, self.state.soft.shape[-1])
            embeds[index] = self.model.self_conditioning(embeds[index], soft.to(embeds.dtype))
            return embeds
        for slot, start, width in zip(slots_np.tolist(), starts.tolist(), widths.tolist()):
            span = embeds[start : start + width]
            soft = self.state.soft[slot, :width].to(embeds.dtype)
            span.copy_(self.model.self_conditioning(span, soft))
        return embeds

    def prepare_inputs(self, input_batch, req_states) -> dict[str, Any]:
        num_tokens = input_batch.num_tokens
        padded = input_batch.num_tokens_after_padding
        positions = self._fill_positions(input_batch)
        if getattr(self.model, "block_routers", ()):
            ids = self._block_ids[:padded]
            ids[:num_tokens] = (
                self._rows(input_batch) * self.max_blocks
                + positions[:num_tokens] // self.state.block_length
            )
            ids[num_tokens:] = self.num_route_blocks - 1
            self._set_blocks(ids)
        if self._inputs_embeds is None:
            return {"positions": positions}
        return {"positions": positions, "inputs_embeds": self._condition(input_batch)}

    def prepare_inputs_embeds(self, scheduled_encoder_inputs, input_batch, req_states):
        if not self.supports_mm_inputs or self._inputs_embeds is None:
            return None
        self.execute_mm_encoder(scheduled_encoder_inputs)
        mm_embeds, is_mm_embed = self.gather_mm_embeddings(input_batch)
        self._mm_embeds = (mm_embeds, is_mm_embed) if mm_embeds else None
        return None

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        positions = self._positions[:num_tokens]
        positions.copy_(torch.arange(num_tokens, device=self.device))
        ids = self._block_ids[:num_tokens]
        ids.copy_((positions // self.state.block_length).clamp_(max=self.num_route_blocks - 1))
        self._set_blocks(ids)
        if self._inputs_embeds is None:
            return {"positions": positions}
        return {"positions": positions, "inputs_embeds": self._inputs_embeds[:num_tokens]}

    def postprocess_state(self, idx_mapping, num_sampled, num_computed_tokens=None):
        return None

    def _remap_slots(self, input_batch, block_tables, slot_mappings, kv_cache_config):
        num_tokens = input_batch.num_tokens
        rows = self._rows(input_batch)
        shifted = self._token_shift(input_batch) > 0
        if self.prune:
            shifted |= self._pruned(input_batch)[rows]
        positions = self._positions[:num_tokens]
        for group, table in enumerate(block_tables):
            size = kv_cache_config.kv_cache_groups[group].kv_cache_spec.block_size
            slots = table[rows, positions // size] * size + positions % size
            current = slot_mappings[group, :num_tokens]
            current.copy_(torch.where(shifted, slots.to(current.dtype), current))

    def _doc_ranges(self, input_batch) -> dict[int, list[tuple[int, int]]] | None:
        if not self.aligned or not self.causal_prefill:
            return None
        num_reqs = input_batch.num_reqs
        num_logits = np.diff(input_batch.cu_num_logits_np[: num_reqs + 1])
        query_lens = np.diff(input_batch.query_start_loc_np[: num_reqs + 1])
        computed = input_batch.num_computed_prefill_tokens_np
        ranges = {
            int(row): prefill_block_ranges(
                int(computed[row] + query_lens[row]), self.state.block_length
            )
            for row in np.where(num_logits == 0)[0]
        }
        return ranges or None

    def prepare_attn(
        self,
        input_batch,
        cudagraph_mode,
        block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
        for_capture=False,
        ubatch_idx: int = 0,
    ) -> dict[str, Any]:
        actual = input_batch.num_reqs
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs, num_tokens = actual, input_batch.num_tokens
        slots = input_batch.idx_mapping[:actual]
        causal = self.state.prefilling[slots]
        if not self.causal_prefill:
            causal = torch.zeros_like(causal)
        if self.causal_commit:
            causal = causal | self.state.committing[slots]
        self._causal_buf[:actual].copy_(causal)
        self._causal_buf[actual:num_reqs] = 0
        seq_lens = self._seq_lens[:num_reqs]
        seq_lens.copy_(input_batch.seq_lens[:num_reqs])
        seq_lens[:actual] -= self.state.shift[slots].to(seq_lens.dtype)
        if self.prune and not for_capture:
            extra = self.state.width[slots] - self._num_logits(input_batch)
            seq_lens[:actual] += torch.where(self._pruned(input_batch), extra, 0).to(
                seq_lens.dtype
            )
        if not for_capture:
            self._fill_positions(input_batch)
            self._remap_slots(input_batch, block_tables, slot_mappings, kv_cache_config)
        ranges = None if for_capture else self._doc_ranges(input_batch)
        bound = None
        if ranges:
            bound = torch.zeros(num_reqs, dtype=torch.int32)
            bound[:actual] = torch.from_numpy(
                input_batch.num_computed_prefill_tokens_np[:actual]
                + np.diff(input_batch.query_start_loc_np[: actual + 1])
            )
        return build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=torch.from_numpy(input_batch.query_start_loc_np),
            max_query_len=input_batch.num_scheduled_tokens.max().item(),
            seq_lens=seq_lens,
            max_seq_len=self.max_model_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            causal=self._causal_buf[:num_reqs],
            mm_req_doc_ranges=ranges,
            seq_lens_cpu_upper_bound=bound,
        )
