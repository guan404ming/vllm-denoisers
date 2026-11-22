# Resident/active scheduling

```bash
DENOISERS_HOLD=sjf DENOISERS_ACTIVE=16 VLLM_PLUGINS=denoisers vllm serve ... \
  --scheduler-cls denoisers.scheduler.resident_active.DenoisersAsyncScheduler
```

- `DENOISERS_HOLD`: `sjf`, `random`, `aging` (with `DENOISERS_AGE_S`), or a hold policy class; see [extending.md](extending.md).
- `DENOISERS_ACTIVE`: how many decoding requests run per step; the rest wait.
- The probe scores each request by the mean entropy of its current block. DiffusionGemma uses a step-count predictor on the same entropies.
- Scores reach the scheduler through a memmap table in the temp dir, keyed by instance id and data-parallel rank. Worker rank 0 writes, the scheduler reads, so tensor-parallel ranks must share one node.
- The denoisers scheduler also aligns prefill chunks to the canvas, which enables chunked prefill for the block-aligned models.

SJF raises SLO20 on DiffusionGemma at 8 and 16 req/s for about 12% lower throughput; see [results.md](results.md). It loses at 4 req/s because it holds requests even when the server is not overloaded.

A custom scheduler declares `aligns_denoiser_prefill` and `uses_denoiser_scores` as class attributes; the denoisers schedulers set both.
