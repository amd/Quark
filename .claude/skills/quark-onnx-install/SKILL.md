---
name: quark-onnx-install
description: >
  Install or verify the correct ONNX Runtime build (and the matching `onnx` package) for a user's
  accelerator backend before Quark ONNX-flow usage. Trigger for "install onnxruntime",
  "pip install onnxruntime", "set up onnxruntime for ROCm", "set up onnxruntime for CUDA",
  "onnxruntime-gpu vs onnxruntime", "onnx version mismatch", "CPU-only onnxruntime installed",
  "onnxruntime providers list missing CUDAExecutionProvider/ROCMExecutionProvider", failures importing
  `onnxruntime`, or any request to get the correct ONNX Runtime build running. Also trigger when
  quark-install reports that ONNX Runtime is missing or mismatched before proceeding with the
  ONNX-to-ONNX flow.
---

Read and follow the instructions in `.claude/skills-impl/l1-atomic/onnx/quark-onnx-install/SKILL.md`.
