import torch

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


def fp8_supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)


def _quantize(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = values.abs().amax(dim=1, keepdim=True).float().clamp(min=1e-12) / FP8_MAX
    return (values / scale).to(torch.float8_e4m3fn), scale


class Fp8Head:
    def __init__(self, weight: torch.Tensor):
        self.weight, scale = _quantize(weight)
        self.scale = scale.t().contiguous()

    def apply(self, layer, x: torch.Tensor, bias=None) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        if flat.shape[0] == 0:
            return flat.new_empty(*x.shape[:-1], self.weight.shape[0])
        quant, scale = _quantize(flat)
        out = torch._scaled_mm(
            quant, self.weight.t(), scale_a=scale, scale_b=self.scale, out_dtype=x.dtype
        )
        return out.reshape(*x.shape[:-1], out.shape[-1])
