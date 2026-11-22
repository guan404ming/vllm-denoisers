# vllm-denoisers

vLLM plugin for diffusion language models. Install it and vLLM loads it; no fork.

| Family | Checkpoints | Kind | Default policy |
|---|---|---|---|
| LLaDA2.0 / 2.1 / 2.2 | `inclusionAI/LLaDA2.x-mini`, `-flash` | block diffusion | `low_confidence` / `joint` / `indel` |
| Stable-DiffCoder | `ByteDance-Seed/Stable-DiffCoder-8B-{Base,Instruct}` | block diffusion | `low_confidence` |
| SDAR, TraDo | `JetLM/SDAR-*-Chat`, `Gen-Verse/TraDo-*` | block diffusion | `low_confidence` |
| Fast-dLLM v2 | `Efficient-Large-Model/Fast_dLLM_v2_*` | block diffusion | `fast_dllm` |
| LLaDA-8B | `GSAI-ML/LLaDA-8B-Instruct` | full attention, decoded per canvas | `llada` |
| Dream | `Dream-org/Dream-v0-Instruct-7B` | full attention, decoded per canvas | `dream` |
| Nemotron Labs Sigma | NVIDIA 3B and 8B, base, instruct, distilled, self-speculative | continuous diffusion | [checkpoint sampler](docs/models.md#nemotron-labs-sigma) |
| DiffusionGemma | `google/diffusiongemma-26B-A4B-it` | uniform noise, self-conditioning | `entropy_bound` |

- Continuous batching, CUDA graphs, prefix caching, tensor parallelism, fp8.
- Pluggable decode, KV, input, and hold policies, per-request decode policies, and model families.
- DiffusionGemma: +20-25% throughput over vLLM's native path on H200, same GSM8K and MMLU.
- Tracks vLLM main; last verified at `5689c2790`.

```bash
uv sync
vllm serve inclusionAI/LLaDA2.0-mini \
  --attention-backend TRITON_ATTN --block-size 32 --no-enable-chunked-prefill
```

## Docs

- [models.md](docs/models.md): per-family flags and decode knobs.
- [deviations.md](docs/deviations.md): where decoding differs from the HF reference.
- [limits.md](docs/limits.md): what is not supported.
- [extending.md](docs/extending.md): custom policies and model families.
- [scheduler.md](docs/scheduler.md): resident/active scheduling.
- [diffusion_gemma.md](docs/diffusion_gemma.md): the DiffusionGemma sampler.
- [results.md](docs/results.md): throughput, SLO, and accuracy.
- [checks.md](docs/checks.md): tests and vLLM coupling points.
