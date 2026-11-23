# Quick start

[Install the plugin](installation.md), then use a GPU with enough memory for `inclusionAI/LLaDA2.0-mini`.

## Start the server

```bash
vllm serve inclusionAI/LLaDA2.0-mini \
  --attention-backend TRITON_ATTN --block-size 32 \
  --max-model-len 2048 --max-num-seqs 8 --max-num-batched-tokens 2048 \
  --generation-config vllm
```

LLaDA2 refines 32 positions at a time. A cache block of 32 keeps prefix caching aligned with the canvas. Wait for the API server to accept connections on port 8000.

## Send a request

Run this command in a second terminal.

```bash
curl --fail-with-body http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"inclusionAI/LLaDA2.0-mini","prompt":"The capital of France is","max_tokens":32}'
```

A successful response contains generated text in `choices[0].text`. The exact text can vary. See the [model guide](models.md) to run a different checkpoint.
