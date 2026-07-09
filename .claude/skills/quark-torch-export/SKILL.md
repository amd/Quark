---
name: quark-torch-export
description: >
  Prepare export and downstream evaluation handoff for a planned or completed Quark Torch PTQ run. Input is a
  PyTorch / HuggingFace transformers model; output formats include HF safetensors, GGUF, and ONNX. Trigger for
  "export model", "save quantized model", "convert to GGUF", "export to HuggingFace format", "export to ONNX",
  or when the user has a completed or planned Torch PTQ run and needs deployment outputs. Not for exporting from
  an .onnx input model — use quark-onnx-export.
---

Read and follow the instructions in `.claude/skills-impl/l1-atomic/torch/quark-torch-export/SKILL.md`.
