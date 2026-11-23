import numpy as np
import torch
from vllm.v1.worker.gpu.model_states.default import DefaultModelState

from denoisers.runtime.sampler import CanvasSampler


class SigmaSampler(CanvasSampler):
    def __init__(self, sampler, max_num_reqs, block, device):
        super().__init__(max_num_reqs, block, device)
        self.penalties_state = sampler.penalties_state

    def add_request(self, req_idx: int, sampling_params) -> None:
        pass

    def apply_staged_writes(self) -> None:
        pass

    def _decode(self, tokens, input_batch, rows, widths, sampled, num_sampled):
        counts = input_batch.cu_num_logits.diff()[: input_batch.num_reqs]
        row = torch.arange(input_batch.num_reqs, device=tokens.device).repeat_interleave(
            counts, output_size=tokens.numel()
        )
        sampled[row, input_batch.expanded_local_pos.long()] = tokens.flatten()
        complete = sampled[:, 0] >= 0
        num_sampled.copy_(torch.where(complete, counts, 0))
        sampled.clamp_min_(0)


class SigmaModelState(DefaultModelState):
    supports_prompt_embeds = False
    num_new_sampled_tokens_per_step = 0

    def add_request(self, req_index, new_req_data):
        super().add_request(req_index, new_req_data)
        self.model.pending.pop(new_req_data.req_id, None)

    def remove_request(self, req_id):
        self.model.pending.pop(req_id, None)
        super().remove_request(req_id)

    def prepare_inputs(self, input_batch, req_states):
        blocks = []
        if input_batch.num_draft_tokens:
            for row, width in enumerate(np.diff(input_batch.cu_num_logits_np)):
                if width:
                    slot = input_batch.idx_mapping_np[row]
                    end = input_batch.query_start_loc_np[row + 1]
                    length = input_batch.num_computed_tokens_np[row]
                    prefix = req_states.all_token_ids.gpu[slot : slot + 1, :length]
                    blocks.append((input_batch.req_ids[row], end - width, width, prefix))
        return {"blocks": blocks}

    def compute_sample_logits(self, hidden_states):
        return hidden_states

    def custom_sampler(self, sampler):
        return SigmaSampler(sampler, self.max_num_reqs, self.model.block, self.device), None

    def dummy_sampler_run(self, sampler, hidden_states):
        return True
