import torch
from vllm.model_executor.models.mistral import MistralForCausalLM
from vllm.model_executor.models.utils import WeightsMapper

from denoisers.models.mixin import BlockDiffusionModel
from denoisers.runtime.model_state import BlockDiffusionModelState


class NemotronDiffusionState(BlockDiffusionModelState):
    def prepare_inputs(self, input_batch, req_states):
        inputs = super().prepare_inputs(input_batch, req_states)
        ends = self.model.prompt_ends
        ends.zero_()
        slots = input_batch.idx_mapping[: input_batch.num_reqs]
        ends[slots] = input_batch.query_start_loc[1 : input_batch.num_reqs + 1].long() - 1
        return inputs

    def prepare_dummy_inputs(self, num_reqs, num_tokens):
        self.model.prompt_ends.zero_()
        return super().prepare_dummy_inputs(num_reqs, num_tokens)


class NemotronLabsDiffusionModel(BlockDiffusionModel, MistralForCausalLM):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_prefix={"encoder.": "model.", "diffusion_head.": "lm_head."}
    )

    def __init__(self, *, vllm_config, prefix=""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        slots = vllm_config.scheduler_config.max_num_seqs
        self.prompt_ends = torch.zeros(slots, dtype=torch.long)
        self.prompt_hidden = torch.empty(
            slots, self.config.hidden_size, dtype=vllm_config.model_config.dtype
        )

    @staticmethod
    def get_model_state_cls():
        return NemotronDiffusionState

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        hidden = super().forward(input_ids, positions, intermediate_tensors, inputs_embeds)
        self.prompt_hidden.copy_(hidden[self.prompt_ends])
        return hidden

    def load_weights(self, weights):
        return super().load_weights(self.hf_to_vllm_mapper.apply(weights))
