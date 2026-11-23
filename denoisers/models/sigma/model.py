import torch
from torch import nn
from transformers import AutoModel, GenerationConfig
from vllm.model_executor.models.utils import AutoWeightsLoader

from denoisers.models.sigma.state import SigmaModelState
from denoisers.models.sigma.steps import SigmaBlock


class NemotronLabsSigmaModel(nn.Module):
    def __init__(self, *, vllm_config, prefix=""):
        super().__init__()
        config = vllm_config.model_config
        self.model = AutoModel.from_config(
            config.hf_config, trust_remote_code=config.trust_remote_code
        )
        self.model.generation_config = GenerationConfig.from_pretrained(
            config.model, revision=config.revision
        )
        self.block = vllm_config.diffusion_config.canvas_length
        self.pending = {}
        self.embeddings = None

    @staticmethod
    def get_model_state_cls():
        return SigmaModelState

    def embed_input_ids(self, input_ids):
        return self.model.encoder.embed_tokens(input_ids)

    def forward(self, input_ids, positions, blocks=(), **kwargs):
        tokens = torch.full_like(positions, -1, dtype=torch.int32)
        for req_id, start, width, prefix in blocks:
            progress = self.pending.get(req_id)
            if progress is None or progress[0] != prefix.shape[1]:
                if self.embeddings is None:
                    self.embeddings = self.model.replaid_embedding_matrix().float()
                steps = SigmaBlock(self.model, prefix.long(), self.embeddings).run()
                self.pending[req_id] = (prefix.shape[1], steps)
            else:
                steps = progress[1]
            result = next(steps)
            if result is not None:
                tokens[start : start + width] = result[:width]
                del self.pending[req_id]
        return tokens.unsqueeze(1)

    def compute_logits(self, hidden_states):
        raise ValueError("Sigma returns sampled tokens directly.")

    def load_weights(self, weights):
        self.embeddings = None
        loaded = AutoWeightsLoader(self.model).load_weights(weights)
        return {f"model.{name}" for name in loaded}
