---
name: quark-env-preflight
description: >
  Collect and normalize environment facts (OS, Python, GPU, CUDA/ROCm, container state) before Quark installation or PTQ planning.
  Trigger for "check my environment", "what GPU do I have", "is my setup ready for Quark", or when any accelerator-related
  assumption is unconfirmed. Also trigger before any install/quantization step where hardware facts are missing.
---

Read and follow the instructions in `.claude/skills-impl/l0-foundation/shared/quark-env-preflight/SKILL.md`.
