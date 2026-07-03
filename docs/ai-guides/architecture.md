# Architecture Overview

Detailed architecture documentation for the Quark codebase.

## Dual Backend Design

Quark has two independent backends that share minimal code:

1. **PyTorch Backend** (`quark/torch/`): Graph-level and eager-mode quantization
2. **ONNX Backend** (`quark/onnx/`): Graph-level quantization of ONNX models

These backends have different capabilities, supported data types, and APIs. Always check which backend is relevant when making changes.

## PyTorch Backend Structure

```text
quark/torch/
├── quantization/       # Core quantization logic
│   ├── api.py         # Main ModelQuantizer API
│   ├── config/        # Configuration classes (Config, QTensorConfig, QLayerConfig)
│   ├── graph/         # FX graph mode quantization
│   ├── observer/      # Calibration observers (MinMax, MSE, Percentile)
│   └── nn/modules/    # Quantized module implementations
├── algorithm/         # Advanced algorithms (AWQ, GPTQ, Qronos, QuaRot)
├── export/            # Export to ONNX, JSON-Safetensors, GGUF
├── pruning/           # Model pruning (layer-wise importance pruning)
└── kernel/            # Custom kernels for ROCm/CUDA
```

## ONNX Backend Structure

```text
quark/onnx/
├── quantization/      # Core quantization logic
│   ├── api.py        # Main ModelQuantizer API (different from PyTorch!)
│   └── config/       # ONNX-specific config classes
├── algorithm/        # AdaQuant, AdaRound, GPTQ, QuaRot, SmoothQuant, CLE
├── operators/        # Custom ONNX operators
├── calibration/      # Calibration methods
└── postprocess/      # Post-quantization refinement
```

## Key Differences: PyTorch vs ONNX

| Aspect | PyTorch Backend | ONNX Backend |
|--------|----------------|--------------|
| Quantization mode | Eager + FX graph | Graph only |
| Group-wise quant | Yes (per-group) | No |
| KV-cache quant | Yes (FP8) | No |
| Algorithms | AWQ, GPTQ, Qronos, QuaRot | AdaQuant, AdaRound, GPTQ, QuaRot, SmoothQuant |
| Export formats | ONNX, JSON-Safetensors, GGUF | N/A (already ONNX) |

## Key File Locations

| Purpose | Path |
|---------|------|
| PyTorch API | `quark/torch/quantization/api.py` |
| ONNX API | `quark/onnx/quantization/api.py` |
| PyTorch Config | `quark/torch/quantization/config/config.py` |
| ONNX Config | `quark/onnx/quantization/config/config.py` |
| PyTorch Examples | `examples/torch/` |
| ONNX Examples | `examples/onnx/` |
| Tests | `test/test_for_torch/`, `test/test_for_onnx/` |
| Tutorials | `tutorials/` |
| Release notes | `docs/source/release_notes.md` |

For usage examples, refer to both `examples/` and `tutorials/` directories rather than documentation snippets.
