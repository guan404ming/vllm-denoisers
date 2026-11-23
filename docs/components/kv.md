# KV policies

Set `kv_policy` in `--hf-overrides` to choose a `KvPolicy`. For masked-position reuse, pass `--hf-overrides '{"kv_policy":"masked"}' --no-async-scheduling`.

| Registered name | Class | Direct base | Behavior |
| --- | --- | --- | --- |
| `masked` | `Masked` | `KvPolicy` | Recompute positions masked before the last update. |
| `window` | `Window` | `KvPolicy` | Recompute the focus window and its neighbors. |

[KV cache reuse](../kv-cache.md)
