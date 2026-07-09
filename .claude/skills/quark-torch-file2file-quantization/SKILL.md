---
name: quark-torch-file2file-quantization
description: >
  Low-memory file2file quantization for very large safetensors LLMs that cannot be loaded whole.
  Use when the user wants to run file2file quantization, adapt a new safetensors checkpoint without
  loading the full model, register an external LLMTemplate, inspect sharded checkpoint naming,
  generate wrapper or conversion scripts, or validate low-memory sharded quantization outputs.
  Trigger for "run file2file quantization", "quantize without loading the model", "large safetensors
  low-memory quantization", "file2file for DeepSeek/Qwen/MoE", "safetensors naming incompatible",
  "register LLMTemplate externally".
---

Read and follow the instructions in `.claude/skills-impl/l1-atomic/torch/quark-torch-file2file-quantization/SKILL.md`.
