# Components

Choose the part of serving you want to change. Each policy setting accepts a registered name or an importable subclass path at server startup.

| Component | What it controls | Setting |
| --- | --- | --- |
| [Schedulers](schedulers.md) | Prefill alignment and request scheduling | `--scheduler-cls` |
| [Decode policies](decode.md) | Which canvas positions to update | `decode_policy` |
| [KV policies](kv.md) | Which cached positions to recompute | `kv_policy` |
| [Input policies](input.md) | Embeddings for the next denoising step | `input_policy` |
| [Hold policies](hold.md) | Which resident requests advance | `DENOISERS_HOLD` |

[Implement a custom policy](../extending.md).
