# vllm-denoisers

vLLM plugin for block-diffusion language models. No fork: install, set `VLLM_PLUGINS=denoisers`, serve.

- LLaDA-8B, LLaDA2.0/2.1/2.2, SDAR, TraDo, Fast-dLLM v2, Dream, and DiffusionGemma on one canvas runtime.
- Continuous batching, CUDA graphs, prefix caching, tensor parallelism, fp8.
- Resident/active scheduler that ranks requests by block entropy.
- DiffusionGemma runs 20-25% faster than vLLM's native path on H200 at the same GSM8K and MMLU scores.
- Tracks vLLM main. Last verified at `5689c2790`.

## Run

```bash
uv sync
VLLM_PLUGINS=denoisers vllm serve inclusionAI/LLaDA2.0-mini \
  --attention-backend TRITON_ATTN --block-size 32 --no-enable-chunked-prefill
```

| Checkpoint | Policy |
|---|---|
| `GSAI-ML/LLaDA-8B-Instruct` | `llada` |
| `inclusionAI/LLaDA2.0-mini`, `-flash` | `low_confidence` |
| `inclusionAI/LLaDA2.1-mini`, `-flash` | `joint` |
| `inclusionAI/LLaDA2.2-mini`, `-flash` | `indel` |
| `JetLM/SDAR-*-Chat`, `JetLM/SDAR-30B-A3B-Chat`, `Gen-Verse/TraDo-*` | `low_confidence` |
| `Efficient-Large-Model/Fast_dLLM_v2_*` | `fast_dllm` |
| `Dream-org/Dream-v0-Instruct-7B` | `dream` |
| `google/diffusiongemma-26B-A4B-it` | `entropy_bound` |

## Docs

- [models.md](docs/models.md): per-family flags and decode knobs.
- [scheduler.md](docs/scheduler.md): resident/active scheduling.
- [diffusion_gemma.md](docs/diffusion_gemma.md): the DiffusionGemma sampler and read extras.
- [deviations.md](docs/deviations.md): where decoding differs from the HF reference.
- [limits.md](docs/limits.md): what is not supported.
- [results.md](docs/results.md): measured throughput, SLO, and accuracy.
- [checks.md](docs/checks.md): tests, parity runs, nightly coupling points.

## Layout

```text
denoisers/
  plugin.py    entry point: configs, tokenizer, models, config hooks
  canvas/      block runtime: state, transition, sampler, ModelState
  policies/    low_confidence, joint, indel, fast_dllm, dream, entropy_bound, kernels
  models/      one folder per family; shared mixin, verify, fp8 head
  scheduler/   probe, ranking policy, vLLM scheduler
patches/       vLLM patch for FLASH_ATTN block-range prefill
tests/         CPU tests; llada2_eval.py runs parity and GSM8K on Modal
docs/
```
