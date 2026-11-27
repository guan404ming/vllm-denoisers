---
icon: model-org
organization: nvidia
---

# [![](../assets/orgs/nvidia.webp){ .org-avatar width="30" height="30" }](https://huggingface.co/nvidia/SDLLM-MDLM-1.7B-Base){ aria-label="SDLLM-MDLM on Hugging Face" target="_blank" rel="noopener" } SDLLM-MDLM

[nvidia/SDLLM-MDLM-1.7B-Base](https://huggingface.co/nvidia/SDLLM-MDLM-1.7B-Base){ target="_blank" rel="noopener" }

## Serve

```bash
vllm serve vllm-denoisers/SDLLM-MDLM-1.7B-Base \
  --no-enable-prefix-caching --max-model-len 2048
```

## Convert

The release ships a training checkpoint, not a Hugging Face folder. [vllm-denoisers/SDLLM-MDLM-1.7B-Base](https://huggingface.co/vllm-denoisers/SDLLM-MDLM-1.7B-Base){ target="_blank" rel="noopener" } holds the converted copy. To build it yourself:

```bash
hf download nvidia/SDLLM-MDLM-1.7B-Base --local-dir SDLLM-MDLM-1.7B-Base
python -m denoisers.models.sdllm.convert SDLLM-MDLM-1.7B-Base SDLLM-MDLM-1.7B-Base-hf
```

## Decoding

| Setting | Value |
| --- | --- |
| Policy | `llada`: sub-block offsets start at the response boundary. |
| Canvas | `canvas_length=256`, `sub_block_length=32` by default. |
| Step budget | `denoise_steps` spans the entire canvas. Set all three options via `--hf-overrides`. |

## Notes

- The converter folds the constant adaLN modulation into the layer norms and output projections. It reads the checkpoint with an unpickler that loads only tensors and containers.
- The release script samples the whole 2048-token sequence with 1,024 ancestral steps. The plugin decodes 256-token canvases with the LLaDA policy, so its outputs differ from `sample.py`.
- This is a base model. Send completion requests without a chat template.
- Prompt text that contains `<s>`, `</s>`, or `<mask>` maps to the special tokens.
- The checkpoint is licensed under CC BY-NC 4.0 for research use only.
