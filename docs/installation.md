# Installation

Use Linux, Python 3.12 or later, and a GPU that vLLM supports, with enough memory for your checkpoint. The plugin uses vLLM worker interfaces that change between releases. Each plugin release supports one vLLM minor version. The command below installs vLLM 0.34.

## Install vLLM and the plugin

Keep the environment active for the quick start.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install 'vllm-denoisers[vllm]'
```

If you already have vLLM 0.34 installed, run `uv pip install vllm-denoisers` in that environment. To work on the plugin, run `uv pip install -e '.[vllm]'` from the root of this repository. For hardware and driver requirements, see the [vLLM installation guide](https://docs.vllm.ai/en/latest/getting_started/installation.html).

## Check the installation

```bash
uv pip show vllm-denoisers
```

vLLM loads installed plugins by default. If you set `VLLM_PLUGINS`, include `denoisers` in the list.

## Choose an attention backend

The default backend is `TRITON_ATTN`. To use FlashAttention, pass `--attention-backend FLASH_ATTN`. It needs FlashAttention 4. Block-aligned prefill on `FLASH_ATTN` also needs a model without sliding-window attention.

Continue with the [quick start](quickstart.md).
