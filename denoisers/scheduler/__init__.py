_BUILTINS = {
    "aligned": "denoisers.scheduler.aligned.AlignedScheduler",
    "elastic": "denoisers.scheduler.elastic.ElasticScheduler",
    "resact": "denoisers.scheduler.resident_active.ResidentActiveScheduler",
}


def aligned_prefill_chunk(start: int, num_new: int, prompt_len: int, block: int) -> int:
    end = start + num_new
    if end >= prompt_len:
        return num_new
    return max(end - end % block - start, 0)


def select_scheduler(scheduler, *, align_prefill=False):
    name = scheduler.scheduler_cls
    if name is None and align_prefill:
        name = "aligned"
    if name in _BUILTINS:
        scheduler.scheduler_cls = _BUILTINS[name]
    elif isinstance(name, str) and "." not in name:
        raise ValueError(
            f"Unknown --scheduler-cls {name!r}. Use aligned, elastic, resact, or a class path."
        )
    elif align_prefill and not getattr(
        scheduler.get_scheduler_cls(), "aligns_denoiser_prefill", False
    ):
        raise ValueError(
            "Chunked prefill needs a scheduler that aligns chunks to the canvas. "
            "Omit --scheduler-cls or pass --no-enable-chunked-prefill."
        )
