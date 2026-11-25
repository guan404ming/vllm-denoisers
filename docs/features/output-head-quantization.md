# Output-head quantization

Set `lm_head_quantization` in `--hf-overrides` for an unquantized output head. For example, `--hf-overrides '{"lm_head_quantization":"fp8"}'`. Shared input embeddings retain their original weights.

| Format | Requirements |
| --- | --- |
| `fp8` | Ada or newer |
| `int4` | Ampere or newer. Input width divisible by 32 and local vocabulary shard size divisible by 8. |
