---
name: quark-onnx-result-validator
description: >
  Validate Quark ONNX quantization output using four lightweight checks:
  auxiliary file copy alignment, expected non-quantized initializer MD5 byte-identity
  (inline `raw_data` + external-data byte ranges), model metadata equality after stripping
  quantization-only opset entries / Quark domains, and fuzzy node-pattern + op-type + dtype
  summaries with QDQ / `com.amd.quark` custom-op presence. Intended for post-quantization
  inspection of `model.onnx` (with or without `model.onnx_data`). Trigger for
  "validate ONNX quantization result", "check quantized .onnx output", "verify ONNX initializers",
  "did QDQ insertion happen", "are the non-quantized weights byte-identical".
---

Read and follow the instructions in `.claude/skills-impl/l1-atomic/onnx/quark-onnx-result-validator/SKILL.md`.
