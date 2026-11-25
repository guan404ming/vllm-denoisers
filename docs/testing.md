# Testing

## Local checks

Run these commands from the repository root.

```bash
uv run pytest -q tests
uv run ruff check .
uv run ruff format --check .
```

The suite skips tests that require an unavailable runtime or GPU. Use a CUDA environment to exercise Triton kernels and quantized output heads.

## Compare with a reference

Use the [quick start](quickstart.md) on a local or hosted GPU. Run the checkpoint’s reference implementation on the same prompts.

### Token parity

Match checkpoint revision, prompt token IDs, decode policy, and output limit. Use greedy decoding. Compare generated token IDs through the first EOS.

[![With identical inputs, the example first differs at the second generated token. Exact parity requires all compared token IDs to match.](assets/token-parity.svg){ width="100%" }](assets/token-parity.svg)

### GSM8K accuracy

Run the first 100 test questions through vLLM and the first 30 through the reference. Use the same chat template and answer instruction. Allow EOS. Score final numeric answers against the gold answers.

[![The plugin runs questions 1 through 100. Both implementations run questions 1 through 30 for the paired accuracy comparison.](assets/paired-accuracy.svg){ width="100%" }](assets/paired-accuracy.svg)

Give vLLM accuracy on all 100, accuracy for each implementation on the paired 30, and the exact token match count for the parity test.

## Check a vLLM upgrade

Verify model registration, attention metadata, the FlashAttention backend, the custom sampler, and the draft token hook against the new worker interfaces. Run a GPU smoke request for each affected family. For DiffusionGemma, also run `uv run pytest -q tests/policies/test_entropy_bound.py` with CUDA available. See the [runtime architecture](architecture.md) for the integration points.
