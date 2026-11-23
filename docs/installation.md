# Installation

Use Linux, Python 3.12 or later, and a CUDA GPU with enough memory for your checkpoint. The plugin uses vLLM worker interfaces that change between builds. The commands below select revision `5689c2790` and the Triton attention backend.

## Install vLLM and the plugin

Run these commands from the root of this repository. Keep the environment active for the quick start.

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install 'vllm==0.33.0rc1.dev487+g5689c2790' \
  --extra-index-url https://wheels.vllm.ai/5689c2790c911a55c75825aac5166305df0b591d/
uv pip install -e .
```

If you already have a compatible vLLM installation, run the last command in that environment. For driver and CUDA requirements, see the [vLLM GPU installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/).

## Check plugin discovery

```bash
python -c 'from importlib.metadata import entry_points
print(entry_points(group="vllm.general_plugins")["denoisers"].value)'
```

The expected output is `denoisers.plugin:register`. vLLM loads installed general plugins by default. If you set `VLLM_PLUGINS`, include `denoisers` in the list.

## Choose an attention backend

Use `TRITON_ATTN` for the shared canvas runtime on stock vLLM. The FlashAttention path on Hopper and newer needs the [block-ranges patch](https://github.com/guan404ming/vllm-denoisers/blob/main/denoisers/patches/vllm-0.33.0rc1.dev487-flash-block-ranges.patch) in the matching vLLM source tree. Per-request causal attention on this path needs FlashAttention 4.

Continue with the [quick start](quickstart.md).
