import numpy as np
import torch
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.sample.output import SamplerOutput


class SigmaSampler:
    def __init__(self, sampler, block):
        self.penalties_state = sampler.penalties_state
        self.block = block

    def add_request(self, slot, params):
        pass

    def apply_staged_writes(self):
        pass

    def __call__(self, tokens, input_batch):
        widths = input_batch.cu_num_logits.diff()
        sampled = tokens.new_zeros((input_batch.num_reqs, self.block))
        rows = torch.arange(input_batch.num_reqs, device=tokens.device).repeat_interleave(
            widths, output_size=tokens.numel()
        )
        sampled[rows, input_batch.expanded_local_pos.long()] = tokens.flatten()
        return SamplerOutput(sampled, None, None, widths, torch.zeros_like(widths))


class SigmaModelState(DefaultModelState):
    supports_prompt_embeds = False
    num_new_sampled_tokens_per_step = 0

    def prepare_inputs(self, input_batch, req_states):
        blocks = []
        if input_batch.num_draft_tokens:
            for row, width in enumerate(np.diff(input_batch.cu_num_logits_np)):
                if width:
                    slot = input_batch.idx_mapping_np[row]
                    end = input_batch.query_start_loc_np[row + 1]
                    length = input_batch.num_computed_tokens_np[row]
                    prefix = req_states.all_token_ids.gpu[slot : slot + 1, :length].long()
                    blocks.append((end - width, int(width), prefix))
        return {"blocks": blocks}

    def compute_sample_logits(self, hidden_states):
        return hidden_states

    def custom_sampler(self, sampler):
        return SigmaSampler(sampler, self.model.block), None

    def dummy_sampler_run(self, sampler, hidden_states):
        return True
