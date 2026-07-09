---
name: quark-torch-result-validator
description: >
  Validate Quark Torch quantization output (HuggingFace safetensors + config.json) using four lightweight checks:
  auxiliary file copy alignment, excluded tensor MD5 byte-identity, config.json deep comparison after stripping
  quantization keys, and safetensors header pattern/dtype summaries. Intended for post-export or file2file
  validation. Trigger for "validate quantization result", "check quantized model output", "verify exported
  weights". Not for .onnx output validation — use quark-onnx-result-validator.
---

Read and follow the instructions in `.claude/skills-impl/l1-atomic/torch/quark-torch-result-validator/SKILL.md`.
