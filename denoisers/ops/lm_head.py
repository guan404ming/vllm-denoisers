from abc import ABC, abstractmethod

import torch

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max
INT4_GROUP_SIZE = 32


class QuantizedHead(ABC):
    @abstractmethod
    def project(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def apply(self, layer, x: torch.Tensor, bias=None) -> torch.Tensor:
        if x.numel() == 0:
            return x.new_empty(*x.shape[:-1], layer.weight.shape[0])
        logits = self.project(x.reshape(-1, x.shape[-1]))
        if bias is not None:
            logits = logits + bias
        return logits.reshape(*x.shape[:-1], logits.shape[-1])


def _quantize_fp8(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = values.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-12) / FP8_MAX
    return (values / scale).to(torch.float8_e4m3fn), scale


class Fp8Head(QuantizedHead):
    def __init__(self, weight: torch.Tensor):
        if not weight.is_cuda or torch.cuda.get_device_capability(weight.device) < (8, 9):
            raise ValueError("fp8 LM head needs CUDA compute capability 8.9 or later.")
        if weight.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("fp8 LM head needs FP16 or BF16 weights.")
        if any(size % 16 for size in weight.shape):
            raise ValueError("fp8 LM head needs both weight dimensions divisible by 16.")
        self.weight, scale = _quantize_fp8(weight)
        self.scale = scale.t().contiguous()

    def project(self, x: torch.Tensor) -> torch.Tensor:
        quant, scale = _quantize_fp8(x)
        return torch._scaled_mm(
            quant, self.weight.t(), scale_a=scale, scale_b=self.scale, out_dtype=x.dtype
        )


class Int4Head(QuantizedHead):
    def __init__(self, weight: torch.Tensor):
        if not weight.is_cuda or torch.cuda.get_device_capability(weight.device) < (8, 0):
            raise ValueError("int4 LM head needs CUDA compute capability 8.0 or later.")
        if weight.dtype != torch.bfloat16:
            raise ValueError("int4 LM head needs BF16 weights and activations.")
        rows, width = weight.shape
        if rows % 8 or width % INT4_GROUP_SIZE:
            raise ValueError(
                "int4 LM head needs shard rows divisible by 8 and width divisible by 32."
            )
        packed = weight.new_empty((rows, width // 2), dtype=torch.uint8)
        self.scales_and_zeros = weight.new_empty((width // INT4_GROUP_SIZE, rows, 2))
        for start in range(0, rows, 1024):
            chunk = weight[start : start + 1024]
            groups = chunk.float().reshape(chunk.shape[0], -1, INT4_GROUP_SIZE)
            low, high = torch.aminmax(groups, dim=-1)
            scale = ((high - low) / 15).clamp_min(1e-12)
            codes = ((groups - low.unsqueeze(-1)) / scale.unsqueeze(-1)).round().to(torch.uint8)
            codes = codes.view_as(chunk)
            packed[start : start + len(chunk)] = (codes[:, ::2] << 4) | codes[:, 1::2]
            params = torch.stack((scale, low + 8 * scale), dim=-1)
            self.scales_and_zeros[:, start : start + len(chunk)] = params.transpose(0, 1)
        self.weight = torch.ops.aten._convert_weight_to_int4pack(packed, 2)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.aten._weight_int4pack_mm(
            x.contiguous(), self.weight, INT4_GROUP_SIZE, self.scales_and_zeros
        )


LM_HEAD_METHODS = {"fp8": Fp8Head, "int4": Int4Head}
