---
name: quark-torch-ptq
description: >
  Torch LLM PTQ workflow for AMD Quark — for PyTorch / HuggingFace transformers models (safetensors
  input). Use when the user wants a complete PTQ pipeline: model inspection, quantization planning, script
  generation, and optional execution. Stops at the quantized output. Trigger for "quantize my model", "run
  PTQ", "run model quantization", "full quantization pipeline", "quantize Llama/Qwen/Mistral with FP8/INT4",
  or any request that spans more than one PTQ step. For a run that also validates and evaluates accuracy use
  quark-torch-llm-ptq-eval. Not for .onnx input models — use quark-onnx-ptq instead.
---

Read and follow the instructions in `.claude/skills-impl/l2-workflows/torch/quark-torch-llm-ptq-workflow/SKILL.md`.
