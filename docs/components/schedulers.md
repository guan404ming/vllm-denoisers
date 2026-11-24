# Schedulers

For automatic selection, omit `--scheduler-cls`. To enable resident/active scheduling, pass `--scheduler-cls resact`.

| Selection | Behavior | Built-in classes (sync / async) |
| --- | --- | --- |
| Automatic | Align prefill chunks when required. Otherwise use vLLM’s default. | Depends on the checkpoint. |
| `aligned` | Align prefill chunks and keep vLLM’s request order. | `AlignedScheduler` / `AlignedAsyncScheduler` |
| `resact` | Choose decode group sizes from measured step times. Rank requests with a hold policy. | `DenoisersScheduler` / `DenoisersAsyncScheduler` |

Both names follow vLLM’s scheduling mode. Add `--no-async-scheduling` for synchronous execution. For a custom scheduler, pass its full class path.

[Serving example](../scheduling.md) · [Hold policies](hold.md)
