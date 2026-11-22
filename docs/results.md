# Results

All numbers: one H200 (NCHC Nano4), vLLM `5689c2790`, FLASH_ATTN, bf16 unless noted. Throughput workloads use 512 prompts x 256 output tokens with `ignore_eos`; "fair" workloads use 500 GSM8K prompts, max 1024 tokens, no `ignore_eos`.

## DiffusionGemma-26B-A4B vs vLLM native

| Batch | vLLM native | denoisers | Gain |
|---|---|---|---|
| 32 | 1962 | 2359 | +20% |
| 64 | 1987 | 2477 | +25% |
| 128 | 1631 | 1735 | +6% |
| 64, TP=2 | 2597 | 3064 | +18% |
| 32, async scheduling | 1910 | 2290 | +20% |

Accuracy: GSM8K 94.8-95.4% (native 94.5%), MMLU 2000 questions 79.2% (native 79.2%), TP=2 GSM8K 94.9%.

Where the gain comes from (per step, batch 32): fused softcap +3%, exact-SC runtime parity +7% total, sparse self-conditioning +10%, fp8 LM head +3%. The native sampler spends about 40% of a step in the LM head, softcap, row stats, and the self-conditioning GEMM.

## SDAR-8B-Chat, canvas 4, threshold 0.9

| Engine | Workload | tok/s |
|---|---|---|
| LMDeploy 0.18 (official engine) | fair, batch 256 | 6397 |
| denoisers | fair, batch 256 | 7158 |
| denoisers | synthetic, batch 256 | 8275 |
| denoisers + fp8 LM head | synthetic, batch 256 | 8839 |
| denoisers + `fp8_per_tensor` | synthetic, batch 256 | 11978 |

With one token a step (`confidence_threshold` 1.01, 128 tokens), the bf16 step takes 7.2 ms at batch 1 to 16 and `fp8_per_tensor` is 1.35-1.37x faster there, 1.28x at batch 64.

## Other models

| Model | Setting | tok/s |
|---|---|---|
| SDAR-1.7B | canvas 4, batch 128 | 16379 |
| SDAR-1.7B | canvas 32, batch 128 | 4654 (fp8 6225) |
| LLaDA2.0-mini | canvas 32, batch 128 | 4709-4841 |
| LLaDA2.0-mini | canvas 32, batch 128, TP=2 | 4112 |

GSM8K 5-shot: SDAR-1.7B 79.4%, LLaDA2.0-mini 91.9%, LLaDA2.2-mini 89.3%.

SDAR-1.7B at canvas 32 with `max_cudagraph_capture_size` 4096 instead of 512: 3711 against 1931 tok/s at batch 32, 4516 against 4351 at batch 128, identical outputs.

## Commit-aware pruning

Throughput: batch 32 or 128 essay prompts x 256 output tokens with `ignore_eos`, `--no-async-scheduling`.

| Model | `prune_refresh` | tok/s batch 32 / 128 | GSM8K |
|---|---|---|---|
| SDAR-1.7B, canvas 32 | off | 1976 / 4346 | |
| SDAR-1.7B, canvas 32 | never | 2341 / 5507 | |
| LLaDA2.0-mini, canvas 32 | off | 1586 / 4526 | 91.9% (strict 77.5%) |
| LLaDA2.0-mini, canvas 32 | never | 1588 / 4319 | 90.2% (strict 77.5%) |
| Fast-dLLM v2-7B, canvas 32 | off | 1171 / 1277 | 79.9% |
| Fast-dLLM v2-7B, canvas 32 | never | 1611 / 2014 | 79.3% |
| Dream-7B, canvas 256 | off | 81 / 81 | 83.5% |
| Dream-7B, canvas 256 | never | 123 / 138 | 76.0% |
| Dream-7B, canvas 256 | 32 | 141 / 101 | 77.5% |
| Dream-7B, canvas 256 | 8 | 131 / 135 | 82.0% |
| Dream-7B, canvas 256 | 4 | 120 / 123 | 82.5% |

LLaDA2.0-mini and Fast-dLLM v2 GSM8K: 1319 questions, 5-shot. On Fast-dLLM v2 the fp8 LM head gives +1% alone and nothing on top of pruning (1645 / 2068 tok/s, GSM8K 80.4%). Dream GSM8K: 200 questions, temperature 0.1, top-p 0.9.

## LLaDA-8B accuracy

GSM8K 0-shot, `gen_length` 512, `block_length` 32, 128 steps, temperature 0.2, answer after `####` else the last number: 67.6% on 500 questions with canvas 768; the HF reference reports about 68%. Canvas 2048 with `max_tokens` 256 gives 30% on 50 questions, because answers sized for the canvas get cut off.

## SLO on DiffusionGemma (192 requests, 3 seeds)

| req/s | Config | SLO30 | SLO20 | p50 s | TTFT p50 s | tok/s |
|---|---|---|---|---|---|---|
| 8 | vLLM native | 0.67 | 0.36 | 25.2 | 14.2 | 2268 |
| 8 | denoisers FCFS | 0.89 | 0.46 | 21.4 | 11.4 | 2503 |
| 8 | denoisers SJF:16 | 0.73 | 0.60 | 16.9 | 10.5 | 2197 |
| 16 | vLLM native | 0.49 | 0.26 | 30.4 | 19.5 | 2259 |
| 16 | denoisers FCFS | 0.57 | 0.31 | 26.7 | 17.1 | 2484 |
| 16 | denoisers SJF:16 | 0.64 | 0.41 | 23.9 | 17.2 | 2222 |

## Scheduling: step cost and the holding trade-off

Step time per active request (ms), measured with `stepcost.py`:

| Active | DiffusionGemma (canvas 256) | LLaDA2.0-mini (32) | SDAR-8B (4) |
|---|---|---|---|
| 1 | 15.8 | 6.0 | 7.6 |
| 16 | 4.6 | 0.51 | 0.50 |
| 64 | 4.15 | 0.53 | 0.18 |
| 256 | | 0.22 | 0.15 |

Holding a request never removes its work, so throughput depends only on per-step efficiency, and the curves keep falling with batch size. A cost-aware budget that holds only beyond the knee (`DENOISERS_EPS` 2%) gave DiffusionGemma +5 pt SLO20 at -2% throughput on 128 requests; a pure value-density ratio collapsed throughput to 1150 tok/s; on SDAR-8B with 960 requests the budget policy lost 12% throughput and SLO. SJF with 16 active remains the strongest SLO setting (+15 pt SLO20 at -12% throughput). The density policy was reverted.

Probe quality (Spearman of the predicted remaining steps against the realized count, 64 requests): DiffusionGemma 0.71 with the refit linear model (mean, median, low fraction, max entropy, step), LLaDA2.0-mini 0.93 from mean block entropy, SDAR-8B 0.63.

## Scheduling on DiffusionGemma, 192 requests, one node, 3 seeds

| Scheduler | req/s | SLO20 | SLO30 | p50 s | p99 s | tok/s |
|---|---|---|---|---|---|---|
| vLLM scheduler, FCFS | 8 | 0.56 | 0.99 | 19.0 | 30.4 | 2727 |
| denoisers scheduler, no hold | 8 | 0.53 | 0.98 | 19.3 | 33.2 | 2682 |
| denoisers SJF:16 | 8 | 0.68 | 0.79 | 12.6 | 49.3 | 2432 |
| vLLM scheduler, FCFS | 16 | 0.40 | 0.72 | 23.3 | 37.9 | 2801 |
| denoisers scheduler, no hold | 16 | 0.37 | 0.66 | 24.3 | 40.0 | 2712 |
| denoisers SJF:16 | 16 | 0.49 | 0.72 | 20.2 | 49.6 | 2453 |

The SJF row pools 12 runs (3 seeds, with and without canvas-aligned prefill chunks, which made no difference). The scheduler class without holding matches vLLM's scheduler within noise and costs about 3% throughput for the probe. Holding to 16 active raises SLO20 by 12 pt at 8 req/s and 9 pt at 16 req/s, lowers p50 by 3-6 s, and costs 11-12% throughput and a longer p99.

## Probe source and aging ablation (SJF:16, 1-2 seeds, SLO20)

| Probe or knob | 8 req/s | 16 req/s | tok/s 8 / 16 |
|---|---|---|---|
| entropy (default) | 0.66 | 0.51 | 2432 / 2571 |
| oracle: recorded steps per block | 0.71 | 0.58 | 2425 / 2491 |
| prompt-length regressor | 0.68 | 0.45 | 2522 / 2578 |
| random | 0.40 | 0.29 | 2517 / 2590 |
| entropy + aging 10 s | 0.62 | 0.46 | 2481 / 2538 |
| entropy + aging 30 s | 0.66 | 0.49 | 2459 / 2536 |

Relative to FCFS, the entropy probe captures 67% (8 req/s) and 64% (16 req/s) of the oracle's SLO20 gain; the prompt-length regressor matches it at 8 req/s but loses at 16 req/s; a random probe is worse than not holding. Aging shortens p99 by 5-10 s at a cost of 0-4 pt SLO20.
