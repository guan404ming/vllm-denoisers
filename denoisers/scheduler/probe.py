import atexit
import hashlib
import os
import tempfile

import numpy as np
import torch

ROW = np.dtype([("key", np.uint64), ("score", np.float32)])


def scheduler_name(scheduler_config) -> str:
    cls = scheduler_config.scheduler_cls
    return cls if isinstance(cls, str) else f"{cls.__module__}.{cls.__name__}"


def uses_denoisers_scheduler(scheduler_config) -> bool:
    return scheduler_name(scheduler_config).startswith("denoisers.scheduler")


def table_path(vllm_config) -> str:
    rank = vllm_config.parallel_config.data_parallel_rank
    return os.path.join(
        tempfile.gettempdir(), f"denoisers-{vllm_config.instance_id}-dp{rank}.scores"
    )


def key(req_id: str) -> int:
    digest = hashlib.blake2b(req_id.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


class ScoreTable:
    def __init__(self, path: str, capacity: int, create: bool = False):
        self.path = path
        self.capacity = capacity
        self.rows = None
        if create:
            self.rows = np.memmap(path, ROW, "w+", shape=(capacity,))
            self.rows["score"] = np.nan
            atexit.register(self.close)

    def write(self, slots, req_ids: list[str], scores: list[float]) -> None:
        self.rows["key"][slots] = [key(rid) for rid in req_ids]
        self.rows["score"][slots] = scores

    def read(self, req_ids: list[str]) -> dict[str, float]:
        if self.rows is None:
            if not os.path.exists(self.path):
                return {}
            self.rows = np.memmap(self.path, ROW, "r", shape=(self.capacity,))
        scored = {
            int(k): float(s)
            for k, s in zip(self.rows["key"], self.rows["score"])
            if not np.isnan(s)
        }
        return {rid: scored[key(rid)] for rid in req_ids if key(rid) in scored}

    def close(self) -> None:
        self.rows = None
        if os.path.exists(self.path):
            os.remove(self.path)


def entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits.float(), dim=-1).clamp(min=-1e4)
    return -(log_probs.exp() * log_probs).sum(dim=-1)


def mean_entropy(entropy: torch.Tensor, counted: torch.Tensor) -> torch.Tensor:
    weight = counted.float()
    return (entropy * weight).sum(dim=1) / weight.sum(dim=1).clamp(min=1)


class Probe:
    def __init__(self, table: ScoreTable):
        self.table = table
        self.pending = None
        self.buffers = [None, None]
        self.ready = None

    def record(
        self, slots, req_ids: list[str], scores: torch.Tensor, committing: torch.Tensor
    ) -> None:
        index = 0
        if self.pending is not None:
            prev_slots, prev_ids, prev_buf, prev_index = self.pending
            if self.ready is not None:
                self.ready.synchronize()
            self.table.write(prev_slots, prev_ids, prev_buf.tolist())
            index = 1 - prev_index
        values = torch.where(committing, torch.nan, scores.float())
        buf = self.buffers[index]
        if buf is None or buf.numel() < values.numel():
            buf = self.buffers[index] = torch.empty(
                values.numel(), dtype=torch.float32, pin_memory=torch.cuda.is_available()
            )
        buf = buf[: values.numel()]
        buf.copy_(values, non_blocking=True)
        if values.is_cuda:
            if self.ready is None:
                self.ready = torch.cuda.Event()
            self.ready.record(torch.cuda.current_stream(values.device))
        self.pending = (slots, req_ids, buf, index)


def uses_probe(scheduler_config) -> bool:
    return scheduler_name(scheduler_config).startswith("denoisers.scheduler.resident_active")


def start(vllm_config) -> Probe:
    capacity = vllm_config.scheduler_config.max_num_seqs
    return Probe(ScoreTable(table_path(vllm_config), capacity, create=True))
