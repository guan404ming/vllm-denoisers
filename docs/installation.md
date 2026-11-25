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

## Check plugin discovery

```bash
python -c 'from importlib.metadata import entry_points
print(entry_points(group="vllm.general_plugins")["denoisers"].value)'
```

The expected output is `denoisers.plugin:register`. vLLM loads installed general plugins by default. If you set `VLLM_PLUGINS`, include `denoisers` in the list.

## Choose an attention backend

Use `TRITON_ATTN` or `FLASH_ATTN` for the shared canvas runtime on stock vLLM. `FLASH_ATTN` needs FlashAttention 4. Block-aligned prefill on `FLASH_ATTN` also needs a model without sliding-window attention.

Continue with the [quick start](quickstart.md).
