import re

from vllm.model_executor.models.llama import LlamaForCausalLM
from vllm.model_executor.models.utils import WeightsMapper

from denoisers.models.mixin import BlockDiffusionModel


class LLaDAModelLM(BlockDiffusionModel, LlamaForCausalLM):
    hf_to_vllm_mapper = WeightsMapper(
        orig_to_new_regex={re.compile(r"^model\.transformer\.ff_out\."): "lm_head."},
        orig_to_new_substr={
            "model.transformer.blocks.": "model.layers.",
            ".attn_norm.": ".input_layernorm.",
            ".ff_norm.": ".post_attention_layernorm.",
            ".attn_out.": ".self_attn.o_proj.",
            ".q_proj.": ".self_attn.q_proj.",
            ".k_proj.": ".self_attn.k_proj.",
            ".v_proj.": ".self_attn.v_proj.",
            ".ff_proj.": ".mlp.gate_proj.",
            ".up_proj.": ".mlp.up_proj.",
            ".ff_out.": ".mlp.down_proj.",
        },
        orig_to_new_prefix={
            "model.transformer.wte.": "model.embed_tokens.",
            "model.transformer.ln_f.": "model.norm.",
        },
    )

    def load_weights(self, weights):
        return super().load_weights(self.hf_to_vllm_mapper.apply(weights))
