import time

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.worker.gpu.cudagraph_utils import CudaGraphManager

logger = init_logger("vllm.denoisers")

_query_lens = VllmConfig.get_decode_cudagraph_query_lens
_capture = CudaGraphManager.capture


def _widths(vllm_config) -> list[int] | None:
    return getattr(getattr(vllm_config.model_config, "hf_config", None), "speculate_widths", None)


def get_decode_cudagraph_query_lens(self, decode_query_len: int) -> dict[int, int]:
    widths = _widths(self)
    if not widths:
        return _query_lens(self, decode_query_len)
    return dict.fromkeys(widths, self.scheduler_config.max_num_seqs)


def capture(self, *args, **kwargs) -> None:
    _capture(self, *args, **kwargs)
    if not _widths(self.vllm_config) or self._max_full_descs_to_capture is not None:
        return
    costs: dict[int, float] = {}
    mixed: dict[int, float] = {}
    for desc, graph in self.graphs.items():
        graph.replay()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        cost = (time.perf_counter() - start) / 3 * 1e3
        table = mixed if desc.uniform_token_count is None else costs
        table[desc.num_tokens] = min(cost, table.get(desc.num_tokens, cost))
    hf_config = self.vllm_config.model_config.hf_config
    hf_config.speculate_costs = sorted((costs or mixed).items())
    logger.info("Probed forward costs (rows, ms): %s", hf_config.speculate_costs)


def apply() -> None:
    VllmConfig.get_decode_cudagraph_query_lens = get_decode_cudagraph_query_lens
    CudaGraphManager.capture = capture
