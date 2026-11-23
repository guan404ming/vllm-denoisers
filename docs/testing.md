# Testing

## Local checks

Run these commands from the repository root.

```bash
uv run pytest -q tests
uv run ruff check .
uv run ruff format --check .
```

The suite skips tests that require an unavailable runtime or GPU. Use a CUDA environment to exercise Triton kernels and quantized output heads.

## Compare with a reference on Modal

The [evaluation driver](../tests/llada2_eval.py) builds separate images for the vLLM and Hugging Face runs. Install the Modal CLI, authenticate, and configure its `huggingface` secret before starting a remote job.

```bash
modal run tests/llada2_eval.py::parity --model inclusionAI/LLaDA2.1-mini --policy joint
modal run tests/llada2_eval.py::features --model JetLM/SDAR-1.7B-Chat
```

Use `--n-ref 0` on the parity command for a vLLM-only smoke test. To compare with Hugging Face, inspect the reported exact token matches. The script prints match counts even when outputs differ. Use a fresh `--cache-dir` when the code or checkpoint changes. The driver pins its own vLLM wheel and reference Transformers version.

## Check a vLLM upgrade

Verify model registration, attention metadata, the custom sampler, and the draft token hook against the new worker interfaces. Run a GPU smoke request for each affected family. For DiffusionGemma, also run `uv run pytest -q tests/test_entropy_bound.py` with CUDA available. See the [runtime architecture](architecture.md) for the integration points.
