# Hold policies

With a resident/active scheduler, set `DENOISERS_HOLD` to choose a `HoldPolicy`. For example, `DENOISERS_HOLD=aging DENOISERS_AGE_S=2` increases priority with wait time. `DENOISERS_LOSS` sets the throughput share the scheduler can give up to hold requests.

| Registered name | Class | Direct base | Behavior |
| --- | --- | --- | --- |
| `fair` | `FairShare` | `HoldPolicy` | Prefer lower scores and credit fair-share steps waited. |
| `sjf` | `ShortestFirst` | `HoldPolicy` | Prefer lower probe scores. |
| `aging` | `Aging` | `HoldPolicy` | Increase priority with time spent waiting. |
| `random` | `RandomOrder` | `HoldPolicy` | Shuffle candidate requests. |

[Request scheduling](../scheduling.md)
