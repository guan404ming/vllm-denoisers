from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm")

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead

from denoisers.models.common.mixin import BlockDiffusionModel


class HeadLoader:
    def load_weights(self, weight):
        self.lm_head.weight.detach().copy_(weight)
        return {"lm_head.weight"}


class Model(BlockDiffusionModel, HeadLoader):
    def __init__(self):
        self.config = SimpleNamespace()
        self.lm_head = ParallelLMHead(64, 32, params_dtype=torch.bfloat16, disable_tp=True)


def test_native_head_and_conflicting_methods():
    model = Model()
    native = model.lm_head.quant_method
    weight = torch.ones_like(model.lm_head.weight)
    assert model.load_weights(weight) == {"lm_head.weight"}
    assert model.lm_head.quant_method is native
    model.config.lm_head_quantization = "unknown"
    with pytest.raises(ValueError, match="Unknown lm_head_quantization"):
        model.load_weights(weight)
    model.config.lm_head_quantization = "int4"
    model.lm_head.quant_method = object()
    with pytest.raises(ValueError, match="unquantized LM head"):
        model.load_weights(weight)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_int4_head_loads_and_refreshes_logits():
    with set_current_vllm_config(VllmConfig()), torch.device("cuda"):
        model = Model()
        model.config.lm_head_quantization = "int4"
        processor = LogitsProcessor(64)
        x = torch.ones(2, 32, dtype=torch.bfloat16)
        for value in (1, 2):
            weight = torch.full_like(model.lm_head.weight, value)
            model.load_weights(weight)
            logits = processor(model.lm_head, x)
            torch.testing.assert_close(logits, torch.nn.functional.linear(x, weight))
