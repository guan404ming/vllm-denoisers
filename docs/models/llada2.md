# LLaDA2

Follow the [quick start](../quickstart.md) to serve `inclusionAI/LLaDA2.0-mini`. The mini and flash checkpoints use a 32-token canvas by default.

| Checkpoint family | Policy | Selection |
| --- | --- | --- |
| LLaDA2.0 | `low_confidence` | Default for checkpoints without expert capacity or a policy override |
| LLaDA2.1 | `joint` | Pass `--hf-overrides '{"decode_policy":"joint"}'` to enable joint token editing |
| LLaDA2.2 | `indel` | The config selects insert/delete decoding when the checkpoint sets expert capacity |

Use `confidence_threshold` for unmasking and `editing_threshold` for edits. The indel policy can split and delete positions. Its anti-loop rule resamples the lowest-confidence changed position. The reference picks a changed position at random.

LLaDA2.2 block routing needs unquantized MoE kernels with expert load balancing disabled. Its fast tokenizer handles plain text. The reference tokenizer's tool rendering and long-text chunking are outside this adapter.
