# Components

Choose the part of serving you want to change. Each policy setting accepts a registered name or an importable subclass path at server startup.

| Component | What it controls | Setting |
| --- | --- | --- |
| [Schedulers](schedulers.md) | Prefill alignment, request scheduling, and [hold policies](schedulers.md#hold-policies) | `--scheduler-cls` |
| [Decode policies](decode.md) | Which canvas positions to update | `decode_policy` |
| [KV policies](kv.md) | Which cached positions to recompute | `kv_policy` |
| [Feedback policies](feedback.md) | Predictions fed into the next denoising step | `feedback_policy` |

[Implement a custom policy](../extending.md).
