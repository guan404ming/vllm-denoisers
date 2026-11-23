# KV cache reuse

Enable a KV policy with `--hf-overrides` and pass `--no-async-scheduling`. Between full passes, the runtime recomputes selected positions and reuses cached keys and values elsewhere.

| Policy | Positions to recompute |
| --- | --- |
| `{"kv_policy":"masked"}` | Positions masked before the previous update. |
| `{"kv_policy":"window"}` | The decoder's focus window, widened by `prefix_look` and `after_look`. The runtime refreshes the canvas when the focus moves. |

- Set `kv_refresh` to a positive step interval for periodic full passes.
- For windowed updates, `llada` and `fast_dllm` expose sub-block focus. Disable prefix caching when the policy refreshes committed prefix tokens. See the [KV policy reference](extending.md#kv-policy).
