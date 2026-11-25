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

The plugin chooses the backend. It uses `FLASH_ATTN` when FlashAttention 4 is available and the model fits, and `TRITON_ATTN` otherwise. To choose one yourself, pass `--attention-backend TRITON_ATTN` or `--attention-backend FLASH_ATTN`. The plugin chooses `TRITON_ATTN` for a model with sliding-window attention and for a decode policy with a causal commit, such as DiffusionGemma.

Continue with the [quick start](quickstart.md).
