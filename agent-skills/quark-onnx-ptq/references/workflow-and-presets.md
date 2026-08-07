# ONNX workflow and preset selection

Pin the Quark release and verify the API against that release. The basic flow is model inspection, calibration reader construction, QConfig selection, `ModelQuantizer`, and validation.

## Environment checks

```bash
python -m pip show amd-quark onnx onnxruntime onnxruntime-gpu onnxruntime-rocm
python -c "import quark, onnx, onnxruntime as ort; print(quark.__version__, onnx.__version__, ort.__version__); print(ort.get_available_providers())"
```

Quark 0.12 documentation requires ONNX Runtime `>=1.22.2,<=1.25.1`. Install only one runtime variant. Current ROCm 7.x guidance may use CPU ONNX Runtime; do not assume a ROCm execution provider exists.

## Minimal static PTQ shape

Adapt this from the [official basic usage guide](https://quark.docs.amd.com/latest/onnx/basic_usage_onnx.html). The data reader is model-specific; the placeholder must not be executed unchanged.

```python
from quark.onnx import ModelQuantizer, QConfig

input_model_path = "/path/to/float.onnx"
quantized_model_path = "/path/to/quantized.onnx"

# Implement an onnxruntime CalibrationDataReader that emits dictionaries
# keyed by the model's actual input names and rewinds deterministically.
calibration_data_reader = build_calibration_reader(input_model_path, calibration_inputs)

# Example only. Select a preset supported by the pinned release and target.
quantization_config = QConfig.get_default_config("A8W8")
quantizer = ModelQuantizer(quantization_config)
quantizer.quantize_model(input_model_path, quantized_model_path, calibration_data_reader)
```

Before execution, print the full generated script and its data paths.

## Preset decision rules

- Use a documented default preset that matches the deployment target before composing a custom config.
- Confirm whether the target accepts QDQ, QOperator, custom `com.amd.quark` operators, wide integer types, BFP, or MX formats.
- INT8 activation/weight PTQ generally requires representative calibration data.
- Weight-only INT4 and dynamic quantization have different calibration and runtime requirements; verify both API and consumer support.
- Ryzen AI/NPU paths may require power-of-two scales, layout conversion, and a target-specific configuration. Follow the current supported-accelerator guide.
- Large models may require ONNX external data. Keep the `.onnx` file and all referenced data together.

## Calibration-reader contract

Validate at least one batch before quantization:

1. keys equal model input names;
2. arrays match expected dtype and rank;
3. dynamic dimensions are resolved consistently;
4. preprocessing matches evaluation and deployment;
5. `rewind()` resets iteration;
6. the reader yields the planned sample count and never silently yields zero.
