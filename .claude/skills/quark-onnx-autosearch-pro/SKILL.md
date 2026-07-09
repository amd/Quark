---
name: quark-onnx-autosearch-pro
description: >
  End-to-end Quark ONNX AutoSearchPro recipe — drives `quark.onnx.AutoSearchPro`
  (Optuna-based hyperparameter search) on a `.onnx` model to find the best
  quantization config (activation/weight spec, calibration method, CLE, AdaRound /
  AdaQuant, FastFinetune params). Use when the user wants to "auto search",
  "tune quantization", "find the best quant config", "sweep AdaRound/AdaQuant",
  "run AutoSearchPro / AutoSearch", "two-stage search", or pick one of the built-in
  presets (`ADVANCED_SEARCH`, `XINT8_SEARCH`, `A8W8_SEARCH`, `A16W8_SEARCH`) for
  their `.onnx` model. Not for HuggingFace / safetensors / PyTorch input models —
  use quark-torch-ptq instead. Not for single-shot ONNX PTQ without search —
  use quark-onnx-ptq.
---

Read and follow the instructions in `.claude/skills-impl/l3-recipes/onnx/quark-onnx-autosearch-pro/SKILL.md`.
