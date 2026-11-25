from typing import Any

import numpy as np
import torch
from vllm.config.compilation import CUDAGraphMode
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.model_states.interface import ModelState

from denoisers.canvas.state import CanvasState
from denoisers.policies import feedback, kv
from denoisers.runtime.batch import (
    finished_prefills,
    free_ranges,
    per_token,
    prefill_block_ranges,
    prefill_segments,
)
from denoisers.runtime.decoders import Decoders
from denoisers.runtime.kv import KvRuntime
from denoisers.runtime.sampler import BlockDiffusionSampler
from denoisers.scheduler import scores


def ranged(backend, policy, model) -> bool:
    if backend != AttentionBackendEnum.FLASH_ATTN or not (policy.aligned and policy.causal_prefill):
        return False
    impls = [getattr(module, "impl", None) for module in model.modules()]
    impls = [impl for impl in impls if getattr(impl, "attn_type", None) == AttentionType.DECODER]
    if policy.causal_commit or any(
        getattr(impl, "vllm_flash_attn_version", None) != 4 or impl.sliding_window[0] >= 0
        for impl in impls
    ):
        raise ValueError(
            "Block-aligned prefill on FLASH_ATTN needs FlashAttention 4, no sliding window, "
            "and a bidirectional commit. Use --attention-backend TRITON_ATTN."
        )
    return True


class BlockDiffusionModelState(ModelState):
    num_new_sampled_tokens_per_step: int = 0

    def __init__(self, vllm_config, model, encoder_cache, device) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        hf_config = self.model_config.hf_config
        policy = model.build_policy()
        probed = scores.uses_probe(vllm_config.scheduler_config)
        policy.dense_scores = probed and not policy.sparse_logits
        self.probe = (
            scores.start(vllm_config) if probed and get_tensor_model_parallel_rank() == 0 else None
        )
        self.aligned = policy.aligned
        self.causal_commit = policy.causal_commit
        self.causal_prefill = policy.causal_prefill
        self.ranged = ranged(vllm_config.attention_config.backend, policy, model)
        name = getattr(hf_config, "feedback_policy", None)
        self.feedback = None
        if name is not None:
            self.feedback = feedback.registry.resolve(name).from_config(hf_config, model)
        hidden_size = None if self.feedback is None else self.feedback.hidden_size
        self.state = CanvasState(
            self.max_num_reqs,
            hf_config.canvas_length,
            hf_config.mask_token_id,
            device,
            hidden_size=hidden_size,
        )
        self.decoders = Decoders(
            policy, lambda cls: cls.from_config(hf_config, model), self.state, self.max_num_reqs
        )
        self.kv = None
        name = getattr(hf_config, "kv_policy", None)
        if name is not None:
            kv_cls = kv.registry.resolve(name)

            def build_kv(decoder):
                kv_policy = kv_cls.from_config(hf_config, decoder)
                if kv_policy.refreshes_prefix and vllm_config.cache_config.enable_prefix_caching:
                    raise ValueError(
                        "A kv policy that refreshes the prefix needs prefix caching off."
                    )
                return kv_policy

            refresh = getattr(hf_config, "kv_refresh", 0)
            self.kv = KvRuntime(build_kv, self.state, policy.shifted, refresh)
            self.kv.for_decoder(policy)
        self.max_blocks = -(-self.max_model_len // hf_config.canvas_length)
        self.num_route_blocks = self.max_num_reqs * self.max_blocks + 1
        self._causal_buf = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=device)
        self._seq_lens = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=device)
        self._positions = torch.zeros(self.max_num_tokens, dtype=torch.int64, device=device)
        self._block_ids = torch.zeros(self.max_num_tokens, dtype=torch.int64, device=device)
        if self.supports_mm_inputs and hidden_size is None:
            hidden_size = self.inputs_embeds_size
        self._inputs_embeds = (
            None
            if hidden_size is None
            else torch.zeros(
                self.max_num_tokens, hidden_size, dtype=self.model_config.dtype, device=device
            )
        )
        self._mm_embeds = None
        self._seed_slots: list[int] = []

    def get_supported_generation_tasks(self):
        return ("generate",)

    def custom_sampler(self, sampler: Any) -> tuple[Any, Any] | None:
        return BlockDiffusionSampler(
            sampler,
            self.state,
            self.decoders,
            self.max_model_len,
            self.model.compute_logits,
            self.model_config.get_vocab_size(),
            kv=self.kv,
            feedback=self.feedback,
            probe=self.probe,
            prefix_caching=self.vllm_config.cache_config.enable_prefix_caching,
        ), None

    def compute_sample_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        count = len(self._seed_slots)
        if not count:
            return hidden_states
        for slot, hidden in zip(self._seed_slots, hidden_states[-count:].split(1), strict=True):
            policy = self.decoders.of(slot)
            logits = self.model.compute_logits(hidden)
            self.state.prefill_seed[slot] = policy.prefill_seed(logits)[0]
        return hidden_states[:-count]

    def dummy_sampler_run(self, sampler: Any, hidden_states: torch.Tensor) -> bool:
        sampler.profile_run(hidden_states)
        return True

    def add_request(self, req_index: int, new_req_data: Any) -> None:
        self.state.reset(req_index)

    def _rows(self, input_batch) -> torch.Tensor:
        return per_token(torch.arange(input_batch.num_reqs, device=self.device), input_batch)

    def _token_shift(self, input_batch) -> torch.Tensor:
        slots = input_batch.idx_mapping[: input_batch.num_reqs]
        return per_token(self.state.shift[slots], input_batch)

    def _num_logits(self, input_batch) -> torch.Tensor:
        cu = input_batch.cu_num_logits[: input_batch.num_reqs + 1]
        return cu[1:] - cu[:-1]

    def _pruned(self, input_batch) -> torch.Tensor:
        slots = input_batch.idx_mapping[: input_batch.num_reqs]
        return self.kv.pruned(slots, self._num_logits(input_batch))

    def _active_offset(self, input_batch) -> torch.Tensor:
        rows = self._rows(input_batch)
        slots = input_batch.idx_mapping[: input_batch.num_reqs]
        local = (
            torch.arange(input_batch.num_tokens, device=self.device)
            - input_batch.query_start_loc[rows]
        )
        return self.kv.offsets(slots, rows, local, self._num_logits(input_batch))

    def _set_blocks(self, ids: torch.Tensor) -> None:
        for router in getattr(self.model, "block_routers", ()):
            router.set_blocks(ids, self.num_route_blocks)

    def _fill_positions(self, input_batch) -> torch.Tensor:
        num_tokens = input_batch.num_tokens
        positions = self._positions[: input_batch.num_tokens_after_padding]
        positions.copy_(input_batch.positions[: positions.shape[0]])
        positions[:num_tokens] -= self._token_shift(input_batch)
        if self.kv is not None:
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
        if self.feedback is None or input_batch.num_draft_tokens == 0:
            return embeds
        num_reqs = input_batch.num_reqs
        num_logits = np.diff(input_batch.cu_num_logits_np[: num_reqs + 1])
        rows = np.where(num_logits > 0)[0]
        block = self.state.block_length
        local = np.arange(block)
        valid = local < num_logits[rows, None]
        index = (input_batch.query_start_loc_np[rows, None] + local)[valid]
        flat = (np.arange(len(rows))[:, None] * block + local)[valid]
        slots = async_tensor_h2d(input_batch.idx_mapping_np[rows], device=self.device)
        cells = self.state.active[slots]
        hidden = self.state.soft.shape[-1]
        soft = self.state.soft[slots].gather(
            1, cells.clamp(min=0).unsqueeze(2).expand(-1, -1, hidden)
        )
        soft = (soft * (cells >= 0).unsqueeze(2)).view(-1, hidden)
        index = async_tensor_h2d(index, device=self.device)
        flat = async_tensor_h2d(flat, device=self.device)
        embeds[index] = self.feedback.condition(embeds[index], soft[flat].to(embeds.dtype))
        return embeds

    def prepare_inputs(self, input_batch, req_states) -> dict[str, Any]:
        num_tokens = input_batch.num_tokens
        padded = input_batch.num_tokens_after_padding
        self._seed_slots.clear()
        if input_batch.has_prefill:
            rows = finished_prefills(input_batch)
            slots = input_batch.idx_mapping_np[rows]
            seeded = self.decoders.flags(slots, "prefill_seed")
            if seeded.any():
                self._seed_slots = slots[seeded].tolist()
                ends = input_batch.query_start_loc_np[rows[seeded] + 1] - 1
                indices = async_tensor_h2d(
                    ends, device=self.device, dtype=input_batch.logits_indices.dtype
                )
                input_batch.logits_indices = torch.cat((input_batch.logits_indices, indices))
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
            return
        self.execute_mm_encoder(scheduled_encoder_inputs)
        outputs, mask = self.gather_mm_embeddings(input_batch)
        match self.model.image_encoder_output:
            case "embeddings":
                self._mm_embeds = (outputs, mask) if outputs else None
            case "token_ids":
                if outputs:
                    indices = mask.nonzero().flatten().to(self.device, non_blocking=True)
                    ids = torch.cat(outputs).flatten().to(input_batch.input_ids.dtype)
                    slots = per_token(input_batch.idx_mapping[: input_batch.num_reqs], input_batch)
                    input_batch.input_ids[indices] = ids
                    tokens = req_states.all_token_ids.gpu
                    tokens[slots[indices], input_batch.positions[indices]] = ids.to(tokens.dtype)
            case other:
                raise ValueError(f"Unknown image_encoder_output: {other!r}")

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
        if self.kv is not None:
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

    def _segments(
        self,
        input_batch,
        ranges,
        seq_lens,
        causal,
        block_tables,
        slot_mappings,
        attn_groups,
        kv_cache_config,
    ) -> dict[str, Any]:
        actual = input_batch.num_reqs
        starts = input_batch.query_start_loc_np[: actual + 1]
        computed = input_batch.num_computed_prefill_tokens_np
        owner, lengths, ends = [], [], []
        for row in range(actual):
            length = int(starts[row + 1] - starts[row])
            if row not in ranges:
                owner.append(row)
                lengths.append(length)
                ends.append(-1)
                continue
            last = int(computed[row])
            for cut in prefill_segments(last, length, self.state.block_length):
                owner.append(row)
                lengths.append(cut - last)
                ends.append(cut)
                last = cut
        owner_gpu = async_tensor_h2d(np.asarray(owner, dtype=np.int64), device=self.device)
        ends_np = np.asarray(ends, dtype=np.int32)
        ends_gpu = async_tensor_h2d(ends_np, device=self.device)
        split = ends_gpu >= 0
        loc = np.zeros(len(owner) + 1, dtype=starts.dtype)
        loc[1:] = np.cumsum(lengths)
        whole = (computed[:actual] + np.diff(starts)).astype(np.int32)[np.asarray(owner)]
        return build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=len(owner),
            num_tokens=input_batch.num_tokens,
            query_start_loc_gpu=async_tensor_h2d(loc, device=self.device),
            query_start_loc_cpu=torch.from_numpy(loc),
            max_query_len=max(lengths),
            seq_lens=torch.where(split, ends_gpu.to(seq_lens.dtype), seq_lens[owner_gpu]),
            max_seq_len=self.max_model_len,
            block_tables=[table[owner_gpu] for table in block_tables],
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            causal=torch.where(split, torch.zeros_like(causal[owner_gpu]), causal[owner_gpu]),
            seq_lens_cpu_upper_bound=torch.from_numpy(np.where(ends_np >= 0, ends_np, whole)),
        )

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
        if self.kv is not None and not for_capture:
            extra = self.state.width[slots] - self._num_logits(input_batch)
            seq_lens[:actual] += torch.where(self._pruned(input_batch), extra, 0).to(seq_lens.dtype)
        if not for_capture:
            self._fill_positions(input_batch)
            self._remap_slots(input_batch, block_tables, slot_mappings, kv_cache_config)
        ranges = None if for_capture else self._doc_ranges(input_batch)
        causal = self._causal_buf[:num_reqs]
        if ranges and not self.ranged and cudagraph_mode != CUDAGraphMode.FULL:
            return self._segments(
                input_batch,
                ranges,
                seq_lens,
                causal,
                block_tables,
                slot_mappings,
                attn_groups,
                kv_cache_config,
            )
        bound = None
        if ranges:
            bound = torch.zeros(num_reqs, dtype=torch.int32)
            bound[:actual] = torch.from_numpy(
                input_batch.num_computed_prefill_tokens_np[:actual]
                + np.diff(input_batch.query_start_loc_np[: actual + 1])
            )
            if self.ranged:
                ranges = free_ranges(ranges, input_batch.is_prefilling_np[:actual])
                causal = True
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
            causal=causal,
            mm_req_doc_ranges=ranges,
            seq_lens_cpu_upper_bound=bound,
        )
