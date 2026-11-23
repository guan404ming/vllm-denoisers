# LLaDA2

## Serve

Run the [quick start](../quickstart.md#start-the-server) with `inclusionAI/LLaDA2.0-mini`.

## Decoding

| Setting | Value |
| --- | --- |
| LLaDA2.0 | `low_confidence` by default without expert capacity or a policy override. |
| LLaDA2.1 | Enable joint editing with `--hf-overrides '{"decode_policy":"joint"}'`. |
| LLaDA2.2 | Expert capacity selects `indel` for splits and deletions. |

## Notes

- Mini and flash default to 32-token canvases. Tune `confidence_threshold` for unmasking and `editing_threshold` for edits.
- Indel loop recovery resamples the lowest-confidence changed position. The reference picks one at random.
- LLaDA2.2 requires unquantized MoE kernels and disabled expert load balancing. Its fast tokenizer supports plain text only (no reference tool rendering or long-text chunking).
- The Uni checkpoint accepts one image per request. See [image understanding](../models.md#image-understanding) for launch settings.
