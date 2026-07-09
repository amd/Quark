---
name: quark-onnx-model-intake
description: >
  Inspect a target ONNX model and prepare metadata for Quark ONNX PTQ planning. Use for `.onnx`
  path validation, opset / IR version detection, input-output shape and dtype discovery, op-type
  histogram, quantizable-op counting, deployment-target compatibility checks (CPU / CUDA / ROCm /
  AMD NPU CNN / AMD NPU Transformer), and risk assessment. Trigger for "analyze my ONNX model",
  "check this onnx model", "what opset is this", "can Quark quantize this .onnx", "is my model
  NPU-compatible", "does my model already have QDQ", "is my model larger than 2 GB", or before any
  ONNX quantization step that needs model facts that are missing.
---

Read and follow the instructions in `.claude/skills-impl/l1-atomic/onnx/quark-onnx-model-intake/SKILL.md`.
