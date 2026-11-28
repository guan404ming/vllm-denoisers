import functools

import cutlass.cute as cute
import torch
from cutlass import Int32
from vllm.config import get_current_vllm_config_or_none
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadataBuilder,
)
from vllm.vllm_flash_attn import flash_attn_varlen_func
from vllm.vllm_flash_attn.cute.utils import scalar_to_ssa


@functools.cache
def tree_mask(block: int):
    @cute.jit
    def load_span(aux_tensors, batch_idx):
        return scalar_to_ssa(aux_tensors[0][batch_idx[0]] * Int32(block), Int32)

    @cute.jit
    def mask_mod(batch_idx, head_idx, q_idx, kv_idx, seqlen_info, aux_tensors):
        ctx = scalar_to_ssa(seqlen_info.seqlen_k - seqlen_info.seqlen_q, Int32)
        size = scalar_to_ssa(Int32(block), Int32)
        one = scalar_to_ssa(Int32(1), Int32)
        span = load_span(aux_tensors, batch_idx)
        fills = span - size
        low = q_idx + ctx - (q_idx + ctx) % size
        own = (kv_idx >= low) & (kv_idx < low + size)
        chain = (q_idx + q_idx < fills) & (kv_idx < low)
        tail = (
            (q_idx + q_idx >= fills)
            & (q_idx < fills + size)
            & (kv_idx + kv_idx < low + low - fills)
        )
        tree = (kv_idx < ctx) | own | chain | tail
        flat = kv_idx < low + size
        return ((span >= one) & tree) | ((span < one) & flat) | (span + size < one)

    return mask_mod


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


class TreeFlashAttentionImpl(FlashAttentionImpl):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = get_current_vllm_config_or_none()
        hf_config = getattr(getattr(config, "model_config", None), "hf_config", None)
        self.block = hf_config.canvas_length if getattr(hf_config, "speculate", False) else 0

    def forward(self, layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs):
        if (
            not self.block
            or attn_metadata is None
            or self.attn_type != AttentionType.DECODER
            or (
                attn_metadata.max_query_len <= self.block
                and not torch.cuda.is_current_stream_capturing()
            )
        ):
            return super().forward(
                layer, query, key, value, kv_cache, attn_metadata, output, *args, **kwargs
            )
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        count = attn_metadata.num_actual_tokens
        flash_attn_varlen_func(
            q=query[:count],
            k=key_cache,
            v=value_cache,
            out=output[:count],
            max_seqlen_q=attn_metadata.max_query_len,
            cu_seqlens_q=attn_metadata.query_start_loc,
            max_seqlen_k=attn_metadata.max_seq_len,
            seqused_k=attn_metadata.seq_lens,
            softmax_scale=self.scale,
            causal=False,
            softcap=self.logits_soft_cap,
            block_table=attn_metadata.block_table,
            fa_version=4,
            s_aux=self.sinks,
            mask_mod=tree_mask(self.block),
            aux_tensors=[attn_metadata.causal],
        )
        return output


class RangedFlashAttentionBackend(FlashAttentionBackend):
    @staticmethod
    def get_builder_cls():
        return RangedFlashAttentionMetadataBuilder

    @staticmethod
    def get_impl_cls():
        return TreeFlashAttentionImpl
