# AutoSearch and validation

## When to use AutoSearch

Use manual/default PTQ first. Escalate only when the measured quality misses an agreed target and multiple configuration choices are plausible.

Before launching a search, define:

- preset or explicit search space;
- calibration reader and optional custom evaluator;
- metric and optimization direction;
- trial count, parallel jobs/devices, wall-time limit, disk estimate, and stop condition;
- persistent output, database, and log paths so a run can be resumed or audited.

Show the full search script and cost estimate, then get explicit confirmation. The [official AutoSearch Pro guide](https://quark.docs.amd.com/latest/onnx/user_guide_auto_search_pro.html) is the authority for the pinned release's API and presets.

Do not compare AutoSearch candidates with different data or preprocessing. If using a built-in L1/L2 proxy, say that it is a proxy rather than task accuracy.

## Validation ladder

1. **Structure:** files exist; external data resolves; protobuf loads.
2. **Schema:** `onnx.checker.check_model` passes.
3. **Quantization evidence:** expected Q/DQ, quantized operators, initializer types, or Quark domains exist.
4. **Metadata/I-O:** expected model inputs and outputs remain compatible.
5. **Runtime load:** the intended ONNX Runtime/provider loads the model without silent fallback.
6. **Smoke inference:** fixed input produces finite outputs of expected shapes.
7. **Quality:** task metric or numerical comparison meets the approved threshold.
8. **Performance:** target-hardware size, memory, and latency meet the goal.

The bundled inspector covers only steps 1-4. State that boundary in every result.

## Recovery clues

| Symptom | Check first |
|---|---|
| Provider missing | Installed `onnxruntime*` variant and `ort.get_available_providers()` |
| Silent CPU fallback | Providers passed to the actual session, not only available providers |
| No calibration data | Reader input names, first batch, rewind, and sample count |
| Model exceeds 2 GB | External-data export and colocated data files |
| Custom op not registered | Quark custom-op build, ABI-compatible ORT, and session registration |
| Accuracy collapse | Float baseline, preprocessing parity, calibration representativeness, excluded nodes, preset/algorithm |
| OOM or disk exhaustion | Batch/sample count, disk cache, worker count, external data, AutoSearch trials |
