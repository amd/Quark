---
name: quark-torch-llm-ptq-eval
description: >
  End-to-end Torch LLM PTQ recipe for AMD Quark — for PyTorch / HuggingFace transformers models
  (safetensors input): quantize, then validate, then evaluate accuracy, in one flow. Delegates the PTQ
  path (intake, planning, manifest, execution) to the quark-torch-ptq workflow, runs mandatory structural
  validation, and runs opt-in accuracy evaluation (perplexity / lm_eval / vLLM-accelerated). Trigger for
  "quantize and validate", "quantize and evaluate", "run PTQ end to end with accuracy check", "full PTQ
  pipeline with validation and eval", or "quantize Llama/Qwen/Mistral with FP8/INT4 and measure accuracy".
  For PTQ only (stop at the quantized output, no validation/eval) use quark-torch-ptq. Not for .onnx input
  models — use quark-onnx-ptq instead.
---

Read and follow the instructions in `.claude/skills-impl/l3-recipes/torch/quark-torch-llm-ptq-eval/SKILL.md`.
