# Runtime architecture

The plugin registers checkpoint architectures through vLLM's `vllm.general_plugins` entry point. Most families share the following path. [Sigma](models/nemotron-labs-sigma.md) supplies a separate model state that calls the checkpoint's sampler.

| Component | Responsibility |
| --- | --- |
| [Family registry](../denoisers/models/families.py) | Connect a checkpoint architecture to its model adapter, config verifier, and optional tokenizer. |
| [Model state](../denoisers/runtime/model_state.py) | Prepare input tokens, embeddings, and attention metadata. Install the canvas sampler in the vLLM worker. |
| [Canvas sampler](../denoisers/runtime/sampler.py) | Receive hidden states, select positions for the LM head, group requests by decoder, and return token counts to vLLM. |
| [Canvas transition](../denoisers/canvas/transition.py) | Apply the policy, track convergence, emit completed tokens, and initialize the next canvas. |
| [Resident/active scheduler](../denoisers/scheduler/resident_active.py) | Read worker scores and defer selected requests until a later step. |

## Follow a decode step

The worker gathers canvas state by request slot. The decode policy returns updated rows and a step result. While a block is still denoising, the sampler reports zero sampled tokens and rejects the scheduled query positions. vLLM can schedule the same block again without advancing the committed prefix.

The KV policy can select a subset of positions for the next forward. The [draft token hook](../denoisers/patches/draft_tokens.py) adjusts the draft width that vLLM schedules for each request. An input policy can also store a soft embedding for the next step. Both act on the same before/after canvas transition.

On commit, the sampler reports the emitted token count to vLLM. Policies that do not need commit logits can skip the output-head projection when the request does not ask for logprobs.
