# Hold policies

With a resident/active scheduler, set `DENOISERS_HOLD` to choose a `HoldPolicy`. For example, `DENOISERS_HOLD=aging DENOISERS_AGE_S=2` increases priority with wait time. ResAct chooses the active limit automatically. Set `DENOISERS_ACTIVE` to a positive integer for a fixed limit. See [Request scheduling](../scheduling.md).

| Registered name | Class | Direct base | Behavior |
| --- | --- | --- | --- |
| `sjf` | `ShortestFirst` | `HoldPolicy` | Prefer lower probe scores. |
| `aging` | `Aging` | `HoldPolicy` | Increase priority with time spent waiting. |
| `random` | `RandomOrder` | `HoldPolicy` | Shuffle candidate requests. |

[Request scheduling](../scheduling.md)
