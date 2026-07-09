---
name: quark-onnx-debug
description: >
  Diagnose failed Quark ONNX installation, calibration, quantization, custom-op compilation, or
  export attempts. Use when the user reports an error, stack trace, invalid artifact, missing
  dependency, ORT execution-provider mismatch, silent CPU fallback, OOM during calibration,
  custom-op load failure (BFPQuantizeDequantize / MXQuantizeDequantize / Extended*), or unexpected
  quantization results from the ONNX flow. Trigger for "Quark ONNX error", "onnxruntime error",
  "quantize_static failed", "calibration crashed", "CUDAExecutionProvider not available",
  "ROCMExecutionProvider not available", "custom op library load failed", "model.onnx larger than
  2GB", "external data not found", "AdaRound diverged", "GPTQ ONNX failed", "QuaRot failed",
  "NPU power-of-2 scale", any Python traceback mentioning quark.onnx / onnxruntime / onnx.
---

Read and follow the instructions in `.claude/skills-impl/l1-atomic/onnx/quark-onnx-debug/SKILL.md`.
