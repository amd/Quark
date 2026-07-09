---
name: quark-torch-debug
description: >
  Diagnose failed Quark Torch PTQ installation, execution, script generation, or export attempts. Use when the
  user reports a torch-side error, stack trace, invalid artifact, missing dependency, CUDA OOM, version mismatch,
  or unexpected PTQ results. Trigger for "Quark error", "PTQ failed", "quantization crashed", "CUDA out of memory",
  "import error", "model loading failed", "wrong results", any Python traceback mentioning
  quark.torch / torch._dynamo / transformers / accelerate. Not for onnxruntime tracebacks — use quark-onnx-debug.
---

Read and follow the instructions in `.claude/skills-impl/l1-atomic/torch/quark-torch-debug/SKILL.md`.
