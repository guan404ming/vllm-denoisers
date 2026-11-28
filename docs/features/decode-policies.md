# Decode policies

At each denoising step, the model scores candidate tokens. The decode policy uses those scores to update the canvas and signals when the block is ready to commit.

## Choose a policy

Each token-canvas model family sets a default policy. To change it for a compatible checkpoint, set `decode_policy` in `--hf-overrides`. For example, LLaDA2.1 supports joint editing:

```bash
vllm serve <model> --hf-overrides '{"decode_policy":"joint"}'
```

| Policy | Behavior |
| --- | --- |
| `low_confidence` | Fill masks by confidence and transfer schedule. |
| `llada` | Unmask within the current sub-block. |
| `fast_dllm` | Seed a block and fill its active sub-block. |
| `dream` | Rank masked positions by entropy. |
| `joint` | Unmask positions and edit prior predictions. |
| `indel` | Split or delete positions during refinement. |
| `entropy_bound` | Accept confident tokens and renoise the canvas. |
| `nemotron_vlm` | Fill masks greedily by confidence. |
| `nemotron_diffusion` | Seed the first position from causal logits. |

A request can select a registered policy by name. The policy must match the attention layout of the default policy. See [request options](../configuration/settings.md#request-options).

## Speculative decoding

Speculation verifies the next denoising step and bets on the next block in the same forward. To turn it on, set `speculate` in `--hf-overrides`:

```bash
vllm serve <model> --hf-overrides '{"speculate":true}'
```

- The policy must score and move a block-aligned canvas with a bidirectional commit, such as `low_confidence`, `llada`, `dream`, or `fast_dllm`.
- Speculation does not support `kv_policy`, `feedback_policy`, `--scheduler-cls resact`, or `--enforce-eager`.
- The plugin runs speculation on `FLASH_ATTN` with FlashAttention 4.
- Requests keep the default policy. The plugin ignores a per-request `decode_policy` and logs a warning.
- The `elastic` scheduler picks the width at each step. See [Request scheduling](request-scheduling.md#choose-a-scheduler).

## Position selection

Confidence measures the probability of the selected token. Entropy measures how spread out the probabilities are across candidates at that position.

Suppose two masks compete for one update and each position has three candidate tokens. With greedy predictions, a confidence ranking picks the first mask (60% versus 55%). An entropy ranking picks the second: almost all its probability lies on just two candidates.

[![Candidate probabilities are 60%, 20%, 20% at the first mask and 55%, 44%, 1% at the second. Confidence favors the first and entropy favors the second.](../assets/position-selection.svg){ width="100%" }](../assets/position-selection.svg)

| Policy | Update rule |
| --- | --- |
| [Confidence-based unmasking](../models/llada2.md#decoding) | Fill high-confidence masks first. The transfer schedule sets how many positions to fill, and the confidence threshold can admit more. |
| [Entropy-ranked unmasking](../models/dream.md#decoding) | Fill positions with the lowest uncertainty first. The denoising schedule sets how many to fill. |
| [Joint editing](../models/llada2.md#decoding) | Fill masks and revise earlier predictions. A separate confidence threshold controls edits. |

## Joint editing

A full canvas can still change. If the model now favors C over B and its confidence clears the editing threshold, the policy replaces B. The revised token stays on the canvas until the block commits. Editing ends when a pass makes no changes or reaches the post-edit step limit.

[![Joint editing revises tentative A B to A C. A later pass makes no edits and the block commits.](../assets/joint-editing.svg){ width="100%" }](../assets/joint-editing.svg)

[Write a decode policy](../extending.md#decode-policy)
