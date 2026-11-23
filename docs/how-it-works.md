# How decoding works

A canvas is a block of token positions that a request refines together. The shared runtime keeps a separate canvas for each request. Token-canvas models refine discrete tokens. Continuous-latent models refine latent states and decode them into tokens.

For token-canvas models:

1. **Prefill.** The model reads the prompt. For block-aligned policies, a partial prompt block shares the first canvas with the start of the response.
2. **Denoise.** The model scores the canvas. The decode policy fills masks, edits tokens, or applies fresh noise. The runtime schedules the same block again while the policy still has work.
3. **Commit.** Once the block is ready, the runtime emits its response tokens and starts the next canvas. A streaming client can receive multiple tokens together. A partial prompt block contributes only its generated portion.

For example, with a canvas of four and a six-token prompt, the first four prompt tokens form a full block. The last two prompt tokens occupy the first two positions of the next block. That canvas can produce two new tokens before the next four-position canvas begins.

A KV policy can reuse keys and values at selected positions between steps. An input policy can feed a weighted token embedding into the next step. The [architecture guide](architecture.md) shows where these decisions enter the runtime.
