---
name: quark-onnx-ptq
description: Plan, run, validate, or debug AMD Quark ONNX-to-ONNX post-training quantization. Use for static or dynamic ONNX quantization, INT8/INT4/BF16/BFP/MX formats, calibration data readers, QConfig and ModelQuantizer, Ryzen AI targets, execution-provider issues, AutoSearch, custom operators, or invalid quantized ONNX artifacts. Covers prerequisites through validation without relying on other skills.
---

# Quantize ONNX Models with AMD Quark

Build a reproducible ONNX PTQ workflow around a pinned model, representative data, target execution provider, and measurable acceptance criteria.

## Workflow

### 1. Intake and inspect

Collect:

- input `.onnx` path, external-data files, model size, opset, and expected input shapes/dtypes;
- deployment target (CPU, CUDA, ROCm, Ryzen AI/NPU, or another runtime) and available execution providers;
- desired format/preset or accuracy, size, and latency goal;
- calibration/evaluation datasets, preprocessing, sample counts, and metric;
- memory, disk, and time budget; output path and overwrite policy.

Inspect the model before planning:

```bash
python scripts/inspect_onnx_artifact.py --quantized-model /path/to/input.onnx
python -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
```

The first command labels an unquantized model as a warning, which is expected during intake.

### 2. Verify prerequisites

Use one compatible environment containing Quark, ONNX, and exactly one ONNX Runtime variant. Read [references/workflow-and-presets.md](references/workflow-and-presets.md) and the release-matched [official ONNX guide](https://quark.docs.amd.com/latest/onnx/basic_usage_onnx.html). If custom operators may be needed, explain first-run compilation and confirm before triggering it.

### 3. Build the plan

Specify:

- pinned Quark release and model checksum/path;
- QConfig preset or explicit config, with target-runtime evidence;
- calibration reader behavior, input names/shapes/dtypes, preprocessing, and sample count;
- execution provider and whether CPU fallback is acceptable;
- algorithms, external-data handling, output path, and evaluation threshold;
- estimated peak memory, disk, trial count, and wall time.

Use basic PTQ first unless there is evidence it misses the accuracy target. AutoSearch is an explicit, budgeted escalation; read [references/autosearch-and-validation.md](references/autosearch-and-validation.md).

### 4. Confirm and execute

Show the generated script or exact command. Get explicit confirmation before package changes, downloads, custom-op compilation, quantization, AutoSearch, or overwriting artifacts. Capture stdout/stderr, environment versions, provider list, configuration, and data provenance.

### 5. Validate

1. Run structural and ONNX checker validation:

   ```bash
   python scripts/inspect_onnx_artifact.py \
     --source-model /path/to/float.onnx \
     --quantized-model /path/to/quantized.onnx \
     --output validation_artifacts.json
   ```

2. Load the quantized artifact with the intended execution provider and confirm the provider actually used.
3. Run deterministic smoke inference.
4. Compare float and quantized outputs/accuracy using identical preprocessing and data.
5. If performance matters, measure size, peak memory, and latency on the target runtime.

Report ONNX checker, quantization-marker, metadata/I-O, provider-load, inference, quality, and performance results separately.

## Rules

- Do not treat provider availability as provider usage; capture both.
- A calibration reader must emit the model's exact input names, shapes, and dtypes and must yield at least one batch.
- Keep calibration and evaluation data distinct in accuracy claims.
- Preserve every external-data file with the model. Never move only the `.onnx` protobuf.
- Do not infer deployment compatibility from an ONNX checker pass or Q/DQ node count.
- Do not start AutoSearch without a trial/time/disk budget and explicit approval.
- On failure, preserve the first causal error. Later cancellation or provider-fallback messages may be secondary.

## Deliverable

Return the pinned inputs, generated config/script, exact command, artifact paths, logs, validation matrix, metric delta, provider evidence, limitations, and smallest recovery step.
