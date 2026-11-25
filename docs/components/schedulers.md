# Schedulers

For automatic selection, omit `--scheduler-cls`. To enable resident/active scheduling, pass `--scheduler-cls resact`.

| Selection | Behavior | Built-in classes (sync / async) |
| --- | --- | --- |
| Automatic | Align prefill chunks when required. Otherwise use vLLM’s default. | Depends on the checkpoint. |
| `aligned` | Align prefill chunks and keep vLLM’s request order. | `AlignedScheduler` / `AlignedAsyncScheduler` |
| `resact` | Apply the active limit and rank resident requests with a hold policy. | `DenoisersScheduler` / `DenoisersAsyncScheduler` |

Both names follow vLLM’s scheduling mode. Add `--no-async-scheduling` for synchronous execution. For a custom scheduler, pass its full class path.

## Hold policies

With a resident/active scheduler, set `DENOISERS_HOLD` to choose a `HoldPolicy`. For example, `DENOISERS_HOLD=aging DENOISERS_AGE_S=2` increases priority with wait time. `DENOISERS_ACTIVE` sets how many resident requests advance on one step. `DENOISERS_RESERVE` gives a share of them to the oldest requests.

| Registered name | Class | Direct base | Behavior |
| --- | --- | --- | --- |
| `sjf` | `ShortestFirst` | `HoldPolicy` | Prefer fewer expected remaining steps. |
| `aging` | `Aging` | `HoldPolicy` | Increase priority with time spent waiting. |
| `random` | `RandomOrder` | `HoldPolicy` | Shuffle candidate requests. |

[Serving example](../scheduling.md)
