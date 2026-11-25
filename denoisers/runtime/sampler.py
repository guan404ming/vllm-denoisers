from types import SimpleNamespace

import numpy as np
import torch
from vllm.distributed import get_tp_group
from vllm.logger import init_logger
from vllm.utils.torch_utils import async_tensor_h2d
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores
from vllm.v1.worker.gpu.sample.output import SamplerOutput
from vllm.v1.worker.gpu.sample.penalties import use_penalty

from denoisers.canvas.state import CanvasState
from denoisers.canvas.transition import advance
from denoisers.hooks import draft_tokens
from denoisers.ops.sampling import exclude_filtered_mask
from denoisers.ops.stats import row_stats
from denoisers.policies.feedback import FeedbackPolicy
from denoisers.runtime.batch import finished_prefills, query_lens
from denoisers.runtime.decoders import Decoders
from denoisers.runtime.kv import KvRuntime, inverse_index
from denoisers.scheduler.scores import Probe

logger = init_logger("vllm.denoisers")

_NO_PENALTIES = SimpleNamespace(output_bin_counts=None)
LOGITS_BYTES = 20


def _output(sampled, num_sampled, num_rejected, logprobs=None) -> SamplerOutput:
    return SamplerOutput(
        sampled_token_ids=sampled,
        logprobs_tensors=logprobs,
        num_nans=None,
        num_sampled=num_sampled,
        num_rejected=num_rejected,
    )


class CanvasSampler:
    def __init__(self, max_num_reqs, block_length, device):
        self._sampled = torch.zeros(max_num_reqs, block_length, dtype=torch.int32, device=device)
        self._num_sampled = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)

    def _finish_prefills(self, input_batch):
        pass

    def _decode(self, hidden, input_batch, rows, widths, sampled, num_sampled):
        raise NotImplementedError

    def __call__(self, hidden, input_batch, draft_logits=None) -> SamplerOutput:
        num_reqs = input_batch.num_reqs
        sampled = self._sampled[:num_reqs]
        num_sampled = self._num_sampled[:num_reqs]
        sampled.zero_()
        num_sampled.zero_()
        self._finish_prefills(input_batch)
        if input_batch.num_draft_tokens == 0:
            return _output(sampled[:, :1], num_sampled, num_sampled)

        widths = np.diff(input_batch.cu_num_logits_np[: num_reqs + 1])
        rows_np = np.where(widths > 0)[0]
        if len(rows_np) == 0:
            return _output(sampled[:, :1], num_sampled, num_sampled)

        logprobs = self._decode(hidden, input_batch, rows_np, widths[rows_np], sampled, num_sampled)
        num_logits = input_batch.cu_num_logits[: num_reqs + 1].diff()
        is_denoise = (num_logits > 0) & (num_sampled == 0)
        rejected = torch.where(is_denoise, query_lens(input_batch), num_logits - num_sampled)
        return _output(sampled, num_sampled, rejected.to(num_sampled.dtype), logprobs)


class BlockDiffusionSampler(CanvasSampler):
    def __init__(
        self,
        sampler,
        state: CanvasState,
        decoders: Decoders,
        max_model_len: int,
        compute_logits,
        vocab_size,
        kv: KvRuntime | None = None,
        feedback: FeedbackPolicy | None = None,
        probe: Probe | None = None,
        prefix_caching: bool = False,
    ):
        self.sampling_states = sampler.sampling_states
        self.logprob_token_ids_state = sampler.logprob_token_ids_state
        self.req_states = sampler.req_states
        self.logits_mode = sampler.logprobs_mode in ("raw_logits", "processed_logits")
        self.state = state
        self.decoders = decoders
        self.max_model_len = max_model_len
        self.compute_logits = compute_logits
        self.vocab_size = vocab_size
        self.stash_logprobs = decoders.default.causal_commit
        self._stash: dict[int, LogprobsTensors] = {}
        self.aligned = decoders.default.aligned
        self.kv = kv
        self.feedback = feedback
        self.probe = probe
        max_num_reqs = state.canvas.shape[0]
        device = state.canvas.device
        super().__init__(max_num_reqs, state.block_length, device)
        self._need = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self._max_seq_len = None if prefix_caching else torch.empty_like(self._need)
        self._need_host = torch.zeros(max_num_reqs, dtype=torch.int32, pin_memory=True)
        self._commit_host = torch.zeros(max_num_reqs, dtype=torch.bool, pin_memory=True)
        self._need_event = torch.cuda.Event()
        self._budget = 0
        if kv is not None:
            draft_tokens.provider = self.draft_widths

    def draft_widths(self, input_batch) -> np.ndarray:
        self._need_event.synchronize()
        return self.kv.widths(input_batch.idx_mapping_np)

    def add_request(self, req_idx: int, sampling_params) -> None:
        if use_penalty(sampling_params):
            logger.warning_once("denoisers ignores repetition, frequency, and presence penalties.")
        if self._max_seq_len is not None:
            self._max_seq_len[req_idx] = int(self.req_states.max_seq_len[req_idx])
        self.sampling_states.add_request(req_idx, sampling_params)
        self.logprob_token_ids_state.add_request(req_idx, sampling_params)
        if self.kv is not None:
            self._need_event.synchronize()
            self.kv.reset(req_idx)
        self.state.stop_at(
            req_idx, {sampling_params.eos_token_id, *(sampling_params.stop_token_ids or ())}
        )
        self.decoders.assign(req_idx, sampling_params).add_request(req_idx, sampling_params)

    def apply_staged_writes(self) -> None:
        self.sampling_states.apply_staged_writes()
        self.logprob_token_ids_state.apply_staged_writes()

    @property
    def penalties_state(self):
        return _NO_PENALTIES

    def _budget_tokens(self, device) -> int:
        if device.type != "cuda":
            return 1 << 30
        free = torch.cuda.mem_get_info(device)[0]
        budget = max(self.state.block_length, free // 2 // (self.vocab_size * LOGITS_BYTES))
        group = get_tp_group()
        if group.world_size > 1:
            shared = torch.tensor(budget, device="cpu")
            torch.distributed.all_reduce(shared, torch.distributed.ReduceOp.MIN, group.cpu_group)
            budget = int(shared)
        return budget

    def profile_run(self, hidden_states) -> None:
        rows = min(hidden_states.shape[0], self._budget_tokens(hidden_states.device))
        torch.empty_like(self.compute_logits(hidden_states[:rows]))

    def _finish_prefills(self, input_batch) -> None:
        self._need_event.synchronize()
        if not input_batch.has_prefill:
            return
        state = self.state
        block = state.block_length
        tokens = self.req_states.all_token_ids.gpu
        for row in finished_prefills(input_batch):
            slot = int(input_batch.idx_mapping_np[row])
            length = int(input_batch.prefill_len_np[row])
            shift = 0
            if self.aligned and self.max_model_len - length >= block:
                shift = length % block
            state.begin(slot, tokens[slot, length - shift : length], length - shift)
            slots = torch.tensor([slot], device=state.canvas.device)
            canvas = state.canvas[slot : slot + 1]
            policy = self.decoders.of(slot)
            state.canvas[slot] = policy.init_canvas(canvas, slots, first=True)[0]
            if policy.prefill_seed is not None:
                state.canvas[slot, shift] = state.prefill_seed[slot]
            self.req_states.draft_tokens[slot, :block] = state.canvas[slot]
            self._need[slot] = block - shift
            self._need_host[slot] = block - shift
            self._commit_host[slot] = False
            if self.kv is not None:
                self.kv.reset(slot)
            self._stash.pop(slot, None)

    def _process(self, logits, expanded, slots_np) -> torch.Tensor:
        top_k, top_p, topp_only = self.sampling_states.get_top_k_top_p(
            expanded, slots_np, logits.shape[-1]
        )
        if top_k is not None or top_p is not None:
            logits = logits.float()
            exclude_filtered_mask(logits, self.state.mask_id, top_k, top_p)
            logits = apply_top_k_top_p(logits, top_k, top_p, has_topp_only_rows=topp_only)
        return logits

    def _logits(
        self, hidden, slots, slots_np, widths, widths_np, full
    ) -> tuple[torch.Tensor, torch.Tensor]:
        block = self.state.block_length
        n = len(widths_np)
        identity = self.kv is None or self.kv.identity(slots_np).all()
        if full and identity and (widths_np == block).all():
            expanded = slots.repeat_interleave(widths, output_size=hidden.shape[0])
            logits = self._process(self.compute_logits(hidden), expanded, slots_np)
            have = torch.ones(n, block, dtype=torch.bool, device=hidden.device)
            return logits.view(n, block, -1), have
        device = hidden.device
        starts = async_tensor_h2d(np.cumsum(widths_np) - widths_np, device=device)
        inv = inverse_index(self.state.active[slots], widths, starts)
        valid = inv >= 0
        if full:
            need, count = valid, int(widths_np.sum())
        else:
            need = valid & (self.state.canvas[slots] == self.state.mask_id)
            count = int(self._need_host[slots_np].sum())
        pos = torch.argsort((~need).view(-1).to(torch.int8), stable=True)[:count]
        index = inv.view(-1)[pos].clamp(min=0)
        expanded = slots.unsqueeze(1).expand(n, block).reshape(-1)[pos]
        computed = self._process(self.compute_logits(hidden[index]), expanded, slots_np)
        logits = torch.empty(n * block, computed.shape[-1], dtype=computed.dtype, device=device)
        if full:
            logits.fill_(-torch.inf)
            logits[:, 0] = 0
        logits[pos] = computed
        return logits.view(n, block, -1), need

    def _scores(self, policy, rows, tokens, slot, max_logprobs, max_ids) -> LogprobsTensors:
        rows = policy.logprob_logits(rows)
        k = rows.shape[0]
        return compute_topk_scores(
            rows,
            max(max_logprobs, 0),
            tokens.long(),
            logprob_token_ids_state=self.logprob_token_ids_state if max_ids else None,
            expanded_idx_mapping=(
                torch.full((k,), slot, dtype=torch.int32, device=rows.device) if max_ids else None
            ),
            max_per_req_token_ids=max_ids,
            logits_mode=self.logits_mode,
        )

    def _logprobs(
        self,
        policy,
        logits,
        before,
        after,
        sampled,
        num_sampled,
        slots_np,
        rows_np,
        max_logprobs,
        max_ids,
    ) -> list[tuple[int, LogprobsTensors]]:
        counts = num_sampled.cpu().numpy()
        shifts = before.shift.cpu().numpy()
        converged = after.committing.cpu().numpy() if self.stash_logprobs else None
        parts = []
        for i in range(len(rows_np)):
            k, shift, slot = int(counts[i]), int(shifts[i]), int(slots_np[i])
            if converged is not None and converged[i]:
                width = int(before.width[i])
                self._stash[slot] = self._scores(
                    policy,
                    logits[i, shift:width].float(),
                    after.canvas[i, shift:width],
                    slot,
                    max_logprobs,
                    max_ids,
                )
            if k == 0:
                continue
            scores = self._stash.pop(slot, None)
            if scores is None:
                rows = logits[i, shift : shift + k].float()
                scores = self._scores(policy, rows, sampled[i, :k], slot, max_logprobs, max_ids)
            parts.append((int(rows_np[i]), scores))
        return parts

    def _commit(self, input_batch, rows_np, sampled, num_sampled) -> None:
        state = self.state
        device = sampled.device
        slots_np = input_batch.idx_mapping_np[rows_np]
        for policy, index in self.decoders.groups(slots_np):
            group_np = rows_np if index is None else rows_np[index]
            rows = async_tensor_h2d(group_np, device=device)
            slots = input_batch.idx_mapping[: input_batch.num_reqs][rows]
            before = state.gather(slots, policy.history_length)
            after, row_sampled, row_num_sampled, _ = advance(before, None, policy, state.mask_id)
            state.scatter(slots, after)
            if self.probe is not None:
                ids = [input_batch.req_ids[i] for i in group_np]
                group_slots = input_batch.idx_mapping_np[group_np]
                self.probe.record(
                    group_slots, ids, before.step, before.committing, policy.probe_stop(after)
                )
            self.req_states.draft_tokens[slots, : state.block_length] = after.canvas
            if self.kv is not None:
                self.kv.restore(slots)
            sampled[rows] = row_sampled.to(sampled.dtype)
            num_sampled[rows] = row_num_sampled.to(num_sampled.dtype)

    def _apply(
        self,
        policy,
        input_batch,
        rows_np,
        rows,
        slots,
        logits,
        have,
        sampled,
        num_sampled,
        logprob_args,
    ) -> list[tuple[int, LogprobsTensors]]:
        state = self.state
        block = state.block_length
        slots_np = input_batch.idx_mapping_np[rows_np]
        before = state.gather(slots, policy.history_length)
        remaining = None
        if (
            policy.fixed_revealed_tokens
            and self._max_seq_len is not None
            and logprob_args is None
            and len(rows_np) == input_batch.num_reqs
        ):
            remaining = self._max_seq_len[slots] - self.req_states.total_len.gpu[slots]
        after, row_sampled, row_num_sampled, entropy = advance(
            before, logits, policy, state.mask_id, remaining
        )
        if self.feedback is not None:
            after.soft = self.feedback.soft(logits, have, before, after)
        state.scatter(slots, after)
        self._need[slots] = (after.canvas == state.mask_id).sum(1, dtype=torch.int32)
        if self.probe is not None:
            if entropy is None:
                entropy = row_stats(logits, state.mask_id).entropy
            scores = policy.probe_score(entropy, before, after)
            self.probe.record(
                slots_np,
                [input_batch.req_ids[i] for i in rows_np],
                scores,
                before.committing,
                policy.probe_stop(after),
            )
        if self.kv is not None:
            full = logprob_args is not None or self.decoders.default.dense_scores
            kv_policy = self.kv.for_decoder(policy)
            active = self.kv.update(kv_policy, slots, before, after, full)
            self.req_states.draft_tokens[slots, :block] = self.kv.draft_tokens(
                kv_policy, after, active, self.req_states.all_token_ids.gpu
            )
        else:
            self.req_states.draft_tokens[slots, :block] = after.canvas
        sampled[rows] = row_sampled.to(sampled.dtype)
        num_sampled[rows] = row_num_sampled.to(num_sampled.dtype)
        if logprob_args is None:
            return []
        return self._logprobs(
            policy,
            logits,
            before,
            after,
            row_sampled,
            row_num_sampled,
            slots_np,
            rows_np,
            *logprob_args,
        )

    def _step(
        self, hidden, input_batch, rows_np, widths_np, sampled, num_sampled, full, logprob_args
    ) -> list[tuple[int, LogprobsTensors]]:
        state = self.state
        block = state.block_length
        device = hidden.device
        slots_np = input_batch.idx_mapping_np[rows_np]
        if logprob_args is None:
            skip = ~self.decoders.flags(slots_np, "commit_logits")
            commit = self._commit_host.numpy()[slots_np] & skip
            if commit.any():
                self._commit(input_batch, rows_np[commit], sampled, num_sampled)
                if commit.all():
                    return []
                keep = ~commit
                starts = np.cumsum(widths_np) - widths_np
                index = np.concatenate(
                    [np.arange(s, s + w) for s, w in zip(starts[keep], widths_np[keep])]
                )
                hidden = hidden[async_tensor_h2d(index, device=device)]
                rows_np, widths_np, slots_np = rows_np[keep], widths_np[keep], slots_np[keep]
        rows = async_tensor_h2d(rows_np, device=device)
        slots = input_batch.idx_mapping[: input_batch.num_reqs][rows]
        widths = async_tensor_h2d(widths_np.astype(np.int64), device=device)
        partial_np = widths_np < block
        if self.kv is not None:
            partial_np &= self.kv.identity(slots_np)
        if partial_np.any():
            partial = async_tensor_h2d(partial_np, device=device)
            state.truncate(slots[partial], widths[partial])
        logits, have = self._logits(hidden, slots, slots_np, widths, widths_np, full)
        parts = []
        for policy, index in self.decoders.groups(slots_np):
            if index is None:
                parts += self._apply(
                    policy,
                    input_batch,
                    rows_np,
                    rows,
                    slots,
                    logits,
                    have,
                    sampled,
                    num_sampled,
                    logprob_args,
                )
                continue
            group = async_tensor_h2d(index, device=device)
            parts += self._apply(
                policy,
                input_batch,
                rows_np[index],
                rows[group],
                slots[group],
                logits[group],
                have[group],
                sampled,
                num_sampled,
                logprob_args,
            )
        return parts

    def _decode(self, hidden, input_batch, rows_np, widths_np, sampled, num_sampled):
        block = self.state.block_length
        device = hidden.device
        ends_np = np.cumsum(widths_np)
        slots_np = input_batch.idx_mapping_np[rows_np]
        max_logprobs = self.sampling_states.max_num_logprobs(slots_np)
        max_ids = self.logprob_token_ids_state.max_num_token_ids(slots_np)
        want_logprobs = max_logprobs >= 0 or max_ids > 0
        logprob_args = (max_logprobs, max_ids) if want_logprobs else None
        sparse = self.decoders.all(slots_np, "sparse_logits")
        full = not sparse or self.decoders.default.dense_scores or want_logprobs
        if 2 * len(rows_np) * block > self._budget:
            self._budget = self._budget_tokens(device)
        per_chunk = max(1, self._budget // block)
        parts = []
        for start in range(0, len(rows_np), per_chunk):
            end = min(start + per_chunk, len(rows_np))
            chunk = hidden[ends_np[start] - widths_np[start] : ends_np[end - 1]]
            parts += self._step(
                chunk,
                input_batch,
                rows_np[start:end],
                widths_np[start:end],
                sampled,
                num_sampled,
                full,
                logprob_args,
            )
        self._need_host.copy_(self._need, non_blocking=True)
        self._commit_host.copy_(self.state.committing, non_blocking=True)
        if self.kv is not None:
            self.kv.sync()
        self._need_event.record()

        logprobs = None
        if parts:
            by_row = dict(parts)
            cu_gen, offset, found = [], 0, []
            for i in range(input_batch.num_reqs):
                cu_gen.append(offset)
                if i in by_row:
                    found.append(by_row[i])
                    offset += by_row[i].logprobs.shape[0]
            logprobs = LogprobsTensors(
                logprob_token_ids=torch.cat([p.logprob_token_ids for p in found]),
                logprobs=torch.cat([p.logprobs for p in found]),
                selected_token_ranks=torch.cat([p.selected_token_ranks for p in found]),
                cu_num_generated_tokens=cu_gen,
            )

        return logprobs
