---
icon: model-org
organization: inclusionAI
---

# [![](../assets/orgs/inclusionAI.webp){ .org-avatar width="30" height="30" }](https://huggingface.co/collections/inclusionAI/llada20){ aria-label="LLaDA2 models on Hugging Face" target="_blank" rel="noopener" } LLaDA2

[inclusionAI/LLaDA2.0](https://huggingface.co/collections/inclusionAI/llada20){ target="_blank" rel="noopener" } · [inclusionAI/LLaDA2.1](https://huggingface.co/collections/inclusionAI/llada21){ target="_blank" rel="noopener" } · [inclusionAI/LLaDA2.2](https://huggingface.co/collections/inclusionAI/llada22){ target="_blank" rel="noopener" }

## Serve

```bash
vllm serve inclusionAI/LLaDA2.0-mini --max-model-len 2048 --generation-config vllm
```

## Image input

```bash
vllm serve inclusionAI/LLaDA2.0-Uni \
  --max-model-len 8192 --max-num-seqs 4 --max-num-batched-tokens 8192 --generation-config vllm
```

- Send `image_url` and `text` items in the user message’s `content` array to `POST /v1/chat/completions`. The model returns text.
- Send one image per request. The visual tokenizer converts image patches to discrete codes.
- Keep the checkpoint’s `image_tokenizer/` directory when serving local weights.

## Tool calling

```bash
vllm serve inclusionAI/LLaDA2.2-mini --max-model-len 4096 --generation-config vllm \
  --enable-auto-tool-choice --tool-call-parser kimi_k2
```

- Send `tools` to `POST /v1/chat/completions`. LLaDA2.2 returns `tool_calls`, and one answer can hold more than one call.
- With `stream`, a tool call arrives when its block is complete.
- A named function and `tool_choice: "required"` need structured outputs. Diffusion models do not support them yet.

## Decoding

| Setting | Value |
| --- | --- |
| LLaDA2.0 | `low_confidence` by default without expert capacity or a policy override. |
| LLaDA2.1 | Enable joint editing with `--hf-overrides '{"decode_policy":"joint"}'`. |
| LLaDA2.2 | Expert capacity selects `indel` for splits and deletions. |

## Notes

- Mini and flash default to 32-token canvases. Tune `confidence_threshold` for unmasking and `editing_threshold` for edits.
- Indel loop recovery resamples the lowest-confidence changed position. The reference picks one at random.
- LLaDA2.2 requires unquantized MoE kernels and disabled expert load balancing. Its tokenizer renders tools as TypeScript-style declarations and splits long text, as the reference tokenizer does.
