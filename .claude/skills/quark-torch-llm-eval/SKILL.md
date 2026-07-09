---
name: quark-torch-llm-eval
description: >-
  End-to-end LLM accuracy evaluation on AMD ROCm (ROCm-only) — docker container setup OR host (no-docker) runtime,
  vLLM/SGLang/ATOM serving, lm-eval / lighteval / evalscope benchmarks.
  Use when the user wants to evaluate, benchmark, or compare an LLM's accuracy.
  Trigger for "evaluate this model", "run gsm8k/mmlu/mmlu_pro/aime/gpqa/hellaswag/arc",
  "test accuracy", "measure perplexity", "compare quantized model accuracy",
  "does this mxfp4 model lose accuracy". For evaluating Quark Agent Skills themselves,
  use quark-torch-eval-runner instead.
---

Read and follow the instructions in `.claude/skills-impl/l1-atomic/torch/quark-torch-llm-eval/SKILL.md`.
