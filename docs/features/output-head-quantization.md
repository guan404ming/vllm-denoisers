# Output-head quantization

Set `lm_head_quantization` in `--hf-overrides` for an unquantized output head. Shared input embeddings retain their original weights.

```bash
vllm serve <model> --hf-overrides '{"lm_head_quantization":"fp8"}'
```

| Format | Requirements |
| --- | --- |
| `fp8` | Ada or newer |
| `int4` | Ampere or newer. Input width divisible by 32 and local vocabulary shard size divisible by 8. |
