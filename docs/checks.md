# Checks

## CPU tests

```bash
uv run pytest -q tests
uv run ruff format --check .
uv run ruff check .
```

GPU-only tests (kernels, quantized LM heads, vLLM parity) skip without CUDA or vLLM.

## Parity and GSM8K on Modal

```bash
modal run tests/llada2_eval.py::parity --model inclusionAI/LLaDA2.1-mini --policy joint
modal run tests/llada2_eval.py::parity --model JetLM/SDAR-1.7B-Chat-b32 --n-ref 0 --canvas 32
DENOISERS_REF_TRANSFORMERS=4.53.1 modal run tests/llada2_eval.py::parity --model Efficient-Large-Model/Fast_dLLM_v2_7B
DENOISERS_GPU=H200:2 modal run tests/llada2_eval.py::parity --model inclusionAI/LLaDA2.2-flash --n-ref 0 --tp 2
modal run tests/llada2_eval.py::gsm8k --n 100 --n-ref 30
modal run tests/llada2_eval.py::features --model JetLM/SDAR-1.7B-Chat
```

- `--n-ref 0` skips the HF reference, which has no KV cache.
- `DENOISERS_GPU` picks the Modal GPU; `DENOISERS_REF_TRANSFORMERS` pins transformers for remote-code checkpoints.
- Modal images lack nvcc: the eval image sets `VLLM_ALLREDUCE_USE_FLASHINFER=0`.

## When the vLLM nightly moves

- `models/diffusion_gemma/model.py` subclasses vLLM's `DiffusionGemmaForConditionalGeneration` and calls `Gemma4Config.verify_and_update_config`. The policy reads `embed_tokens.shard_indices`, `model.normalizer`, and `final_logit_softcapping`.
- The other models subclass vLLM's Qwen2, Qwen3, Qwen3-MoE, and BailingMoeV2 classes.
- The runtime uses the `ModelState` hooks `prepare_inputs`, `prepare_inputs_embeds`, `prepare_attn`, `custom_sampler`, `compute_sample_logits`, and `dummy_sampler_run`.
- `denoisers/patches/draft_tokens.py` wraps `DraftTokensHandler.set_draft_tokens` to cap each request's drafts to its active cells. The runner builds that handler without a `ModelState` reference, so the hook is process-wide.
- Prefix re-encoding reads committed token ids from `req_states.all_token_ids`.
- After a bump: CPU tests, then a GPU smoke per family and `tests/test_entropy_bound.py` on a GPU.
