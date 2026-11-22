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

Effect on DiffusionGemma at 8 req/s (H200, 192 requests): SLO20 goes from 0.46 (FCFS) to 0.60, p50 latency from 21 s to 17 s, at about 12% lower throughput and a longer p99. SJF loses at 4 req/s because it holds requests even when the server is not overloaded.

Custom schedulers inherit `aligns_denoiser_prefill` and `uses_denoiser_scores` from the denoisers scheduler classes. Set these class attributes independently when implementing a scheduler.
