import torch
from torch import nn
from transformers import AutoModel, GenerationConfig
from vllm.model_executor.models.utils import AutoWeightsLoader

from denoisers.models.sigma.state import SigmaModelState


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

    @staticmethod
    def get_model_state_cls():
        return SigmaModelState

    def embed_input_ids(self, input_ids):
        return self.model.encoder.embed_tokens(input_ids)

    def generate_block(self, prefix, width):
        generation = self.model.generation_config
        if getattr(generation, "decoding", None) == "self_speculative":
            ids = self.model.generate_speculative(
                prefix,
                max_new_tokens=width,
                draft_width=generation.draft_width,
                diffusion_calls_per_round=generation.diffusion_calls_per_round,
            )
        else:
            distilled = getattr(self.model.config, "native_family", None) == "phased_dmd"
            ids, _ = self.model.generate(
                prefix,
                max_new_tokens=width,
                block_length=self.block,
                steps=self.model.config.native_nfe if distilled else 32,
                sampler="fresh_x0" if distilled else "ddim",
                time_schedule="log_snr",
                guidance_scale=1.0 if distilled else 3.0,
                score_temp=1.0 if distilled else 0.3,
            )
        return ids[0, prefix.shape[1] :]

    def forward(self, input_ids, positions, blocks=(), **kwargs):
        tokens = torch.zeros_like(positions, dtype=torch.int32)
        for start, width, prefix in blocks:
            tokens[start : start + width] = self.generate_block(prefix, width)
        return tokens.unsqueeze(1)

    def compute_logits(self, hidden_states):
        raise ValueError("Sigma returns sampled tokens directly.")

    def load_weights(self, weights):
        loaded = AutoWeightsLoader(self.model).load_weights(weights)
        return {f"model.{name}" for name in loaded}
