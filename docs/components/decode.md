# Decode policies

Set `decode_policy` in `--hf-overrides` to choose a `DecodePolicy` for a compatible checkpoint. For example, LLaDA2.1 supports `--hf-overrides '{"decode_policy":"joint"}'`.

`ScoreMovePolicy` extends `DecodePolicy` with a shared scoring step. Per-request overrides use a registered name and must match the default policy’s attention layout.

| Registered name | Class | Direct base | Behavior |
| --- | --- | --- | --- |
| `low_confidence` | `LowConfidence` | `ScoreMovePolicy` | Fill masks by confidence and transfer schedule. |
| `llada` | `LLaDA` | `ScoreMovePolicy` | Unmask within the current sub-block. |
| `fast_dllm` | `FastDLLM` | `ScoreMovePolicy` | Seed a block and fill its active sub-block. |
| `dream` | `Dream` | `ScoreMovePolicy` | Rank masked positions by entropy. |
| `joint` | `JointThreshold` | `DecodePolicy` | Unmask positions and edit prior predictions. |
| `indel` | `Indel` | `DecodePolicy` | Split or delete positions during refinement. |
| `entropy_bound` | `EntropyBound` | `DecodePolicy` | Accept confident tokens and renoise the canvas. |
| `nemotron_vlm` | `NemotronVLMDiffusion` | `ScoreMovePolicy` | Fill masks greedily by confidence. |
| `nemotron_diffusion` | `NemotronDiffusion` | `NemotronVLMDiffusion` | Seed the first position from causal logits. |

[Decode policy examples](../decode-policies.md)
