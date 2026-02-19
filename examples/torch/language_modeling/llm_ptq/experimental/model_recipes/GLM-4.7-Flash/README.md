<!--
Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

## GLM-4.7-Flash Offline Quantization (Quark)

### Overview

This directory provides a Quark quantization recipe for `zai-org/GLM-4.7-Flash`.

### Requirements

- **Quark**: 0.11 (or a compatible version)
- **Transformers**: >= 5.0.0

### Quantization strategy

GLM-4.7-Flash parameter names look like:

- Dense (layer 0): `model.layers.0.mlp.(gate_proj|up_proj|down_proj).weight`
- MoE Experts (from layer 1): `model.layers.N.mlp.experts.E.(gate_proj|up_proj|down_proj).weight`
- Router gate: `model.layers.N.mlp.gate.*`
- Attention: `model.layers.N.self_attn.*`

### Usage

#### MXFP4 (MoE only, no kv-cache)

```bash
python3 GLM-4.7-Flash/quantize_glm_4.7_flash.py \
  --input-model-path zai-org/GLM-4.7-Flash \
  --output-quantized-hf-path amd/GLM-4.7-Flash-MXFP4 \
  --multi-gpu \
  --preset mxfp4_moe_only_no_kvcache
```

#### FP8 PTPC (attn + MoE, no kv-cache)

```bash
python3 GLM-4.7-Flash/quantize_glm_4.7_flash.py \
  --input-model-path zai-org/GLM-4.7-Flash \
  --output-quantized-hf-path amd/GLM-4.7-Flash-FP8-PTPC \
  --multi-gpu \
  --preset fp8_ptpc_attn_moe_no_kvcache
```

#### FP8 (per-tensor static, attn + MoE, no kv-cache)

```bash
python3 GLM-4.7-Flash/quantize_glm_4.7_flash.py \
  --input-model-path zai-org/GLM-4.7-Flash \
  --output-quantized-hf-path amd/GLM-4.7-Flash-FP8 \
  --multi-gpu \
  --preset fp8_pertensor_attn_moe_no_kvcache
```

#### INT4 weight-only (attn + MoE + lm_head, no kv-cache)

```bash
python3 GLM-4.7-Flash/quantize_glm_4.7_flash.py \
  --input-model-path zai-org/GLM-4.7-Flash \
  --output-quantized-hf-path amd/GLM-4.7-Flash-INT4-WO-128-AWQ \
  --multi-gpu \
  --preset int4_wo_128_awq_attn_moe_lm_head_no_kvcache
```

#### Notes

- To override exclusions, pass `--exclude_layers ...` explicitly; it takes precedence over preset defaults.
- These presets do not quantize KV cache (i.e., no `--kv_cache_dtype/--kv_cache_quant_scheme`).
- Calibration data source:
  - Default is `--dataset pileval` (same as `quantize_quark.py`).
  - Note: `pileval/wikitext/...` may download via HuggingFace Datasets unless already cached locally.

### Evaluation (optional)

This script can run PPL evaluation directly (via Quark's `ppl_eval`). By default, it does not evaluate.

- Run the Python entry directly and control evaluation flags:

```bash
python3 GLM-4.7-Flash/quantize_glm_4.7_flash.py \
  --input-model-path zai-org/GLM-4.7-Flash \
  --output-quantized-hf-path amd/GLM-4.7-Flash-MXFP4 \
  --preset mxfp4_moe_only_no_kvcache \
  --do_evaluation
```

#### PPL Results (calib: pileval, eval: wikitext2)

| Preset | Description | PPL |
|--------|-------------|-----|
| `mxfp4_moe_only_no_kvcache` | MXFP4 (MoE only, no kv-cache) | 11.0825 |
| `fp8_ptpc_attn_moe_no_kvcache` | FP8 PTPC (attn + MoE, no kv-cache) | 10.6969 |
| `fp8_pertensor_attn_moe_no_kvcache` | FP8 per-tensor static (attn + MoE, no kv-cache) | 11.1582 |
| `int4_wo_128_awq_attn_moe_lm_head_no_kvcache` | INT4 weight-only AWQ (attn + MoE + lm_head, no kv-cache) | 14.9810 |
