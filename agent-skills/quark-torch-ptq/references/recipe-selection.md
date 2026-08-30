# PyTorch recipe selection

Start from the release-matched [LLM PTQ guide](https://quark.docs.amd.com/latest/pytorch/example_quark_torch_llm_ptq.html) and its supported-model table.

## Selection guide

| Goal | Candidate | What must be verified |
|---|---|---|
| Broad accelerator-friendly LLM compression | FP8 | Model-family support, FP8-capable target runtime, KV-cache format if requested |
| CPU/static integer path | INT8 | Calibration representativeness and target operator support |
| Maximum weight-memory reduction | INT4/UINT4 weight-only | Group size, AWQ/GPTQ support, target loader compatibility |
| OCP microscaling experiment | MXFP4/MXFP6 | Model and export support plus consuming runtime support |
| Recover accuracy after basic PTQ | AWQ, GPTQ, SmoothQuant, rotation | Algorithm support, calibration cost, and export restrictions |

Do not infer support from a precision name alone. Check the current guide and `quantize_quark.py --help` for the pinned release.

## Release-0.12-era command shapes

These examples show the planning shape; validate every flag against the pinned script before execution.

```bash
# FP8 plus FP8 KV cache, Hugging Face export
python quantize_quark.py \
  --model_dir /path/to/model \
  --output_dir /path/to/output \
  --quant_scheme fp8 \
  --kv_cache_dtype fp8 \
  --num_calib_data 128 \
  --model_export hf_format

# INT4 weight-only with AWQ
python quantize_quark.py \
  --model_dir /path/to/model \
  --output_dir /path/to/output \
  --quant_scheme int4_wo_128 \
  --num_calib_data 128 \
  --quant_algo awq \
  --dataset pileval_for_awq_benchmark \
  --seq_len 512 \
  --model_export hf_format

# INT8 on CPU
python quantize_quark.py \
  --model_dir /path/to/model \
  --output_dir /path/to/output \
  --quant_scheme int8 \
  --num_calib_data 128 \
  --device cpu \
  --model_export hf_format
```

For multi-GPU placement, inspect the pinned release's `--multi_gpu` choices and check free memory on every visible device. Very large MoE models also require sufficient host virtual-memory mappings and disk space; capture those checks in the plan.
