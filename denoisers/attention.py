import torch
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionMetadataBuilder,
)


class RangedFlashAttentionMetadataBuilder(FlashAttentionMetadataBuilder):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        if vllm_config.diffusion_config is None or self.mm_prefix_query_ranges_np is not None:
            return
        shape = (vllm_config.scheduler_config.max_num_batched_tokens, 2)
        self.mm_prefix_query_ranges_cpu = torch.empty(
            shape, dtype=torch.int32, pin_memory=PIN_MEMORY
        )
        self.mm_prefix_query_ranges_np = self.mm_prefix_query_ranges_cpu.numpy()
        self.mm_prefix_query_ranges_gpu = torch.empty(shape, dtype=torch.int32, device=device)


class RangedFlashAttentionBackend(FlashAttentionBackend):
    @staticmethod
    def get_builder_cls():
        return RangedFlashAttentionMetadataBuilder
