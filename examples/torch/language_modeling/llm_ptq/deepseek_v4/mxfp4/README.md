<!--
Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# DeepSeek-V4 MXFP4 Quantization

Re-quantize the MoE expert weights of the official DeepSeek-V4 checkpoint to
Quark MXFP4 with the file-to-file path, using the shared `quantize_quark.py`
entry point and its built-in `deepseek_v4` template.

The official DeepSeek-V4 checkpoint ships in a mixed quantized wire format:

- attention projections: blockwise FP8 (`float8_e4m3fn`) weights with
  `float8_e8m0fnu` block scales;
- routed / shared MoE experts: MXFP4-packed weights (Int8-packed FP4 codes) with
  `float8_e8m0fnu` block scales.

Quark's file-to-file path recognizes this format, re-quantizes the MoE expert
weights to MXFP4, and keeps every excluded layer (attention, router gate, norms,
embeddings, head, MTP block) in its original checkpoint format. It streams one
safetensors shard at a time, so a single GPU is enough and the full model is
never loaded into device memory.

## Quantization

Run from `examples/torch/language_modeling/llm_ptq/`. The `deepseek_v4` template
is registered in `quantize_quark.py`, and `--file2file_quantization` auto-detects
the model type from `config.json`:

```bash
python quantize_quark.py \
    --model_dir /path/to/DeepSeek-V4-Flash \
    --quant_scheme mxfp4 \
    --file2file_quantization \
    --output_dir /path/to/DeepSeek-V4-Flash-MXFP4
```

## Evaluation

The MXFP4 checkpoint is served with SGLang and evaluated with `lm-eval` on
GSM8K. See [amd/DeepSeek-V4-Flash-MXFP4](https://huggingface.co/amd/DeepSeek-V4-Flash-MXFP4) for the full
evaluation setup (SGLang launch command, `lm_eval` invocation and environment
flags).

| Benchmark | deepseek-ai/DeepSeek-V4-Flash | amd/DeepSeek-V4-Flash-MXFP4 | Recovery |
|---|---|---|---|
| GSM8K (flexible-extract) | 85.14 | 83.78 | 98.4% |
