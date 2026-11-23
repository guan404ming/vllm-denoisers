# KV cache reuse

Enable a KV policy with `--hf-overrides` and pass `--no-async-scheduling`. Between full passes, the runtime recomputes selected positions and reuses cached keys and values elsewhere.

| Policy | Positions to recompute |
| --- | --- |
| `{"kv_policy":"masked"}` | Positions masked before the previous update. |
| `{"kv_policy":"window"}` | The decoder's focus window, widened by `prefix_look` and `after_look`. The runtime refreshes the canvas when the focus moves. |

## Masked updates

[![Positions 2 and 4 start masked. The next partial pass recomputes both, even though position 2 now holds B.](assets/masked-kv-reuse.svg){ width="100%" }](assets/masked-kv-reuse.svg)

## Windowed updates

[![With focus on positions 3 and 4 and one neighbor on each side, the next partial pass recomputes positions 2 through 5.](assets/window-kv-reuse.svg){ width="100%" }](assets/window-kv-reuse.svg)

- Set `kv_refresh` to a positive step interval for periodic full passes.
- For windowed updates, `llada` and `fast_dllm` expose sub-block focus. Disable prefix caching when the policy refreshes committed prefix tokens. See the [KV policy reference](extending.md#kv-policy).
