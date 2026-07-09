---
name: quark-onnx-ptq
description: >
  End-to-end ONNX PTQ workflow for AMD Quark — for `.onnx` input models (with optional sibling
  `.onnx_data` external-weights file). Use when the user wants a complete ONNX-to-ONNX pipeline:
  model intake, quantization planning, calibration-script generation, manifest, and confirmed
  execution. Trigger for "quantize my .onnx", "run ONNX PTQ end to end",
  "full ONNX quantization pipeline", "quantize yolov8/resnet50/yolo_nas with XINT8/A8W8/BFP16/MXFP*",
  "weights-only INT4 for my .onnx LLM", or any request that spans more than one ONNX PTQ step.
  Not for HuggingFace / safetensors / PyTorch input models — use quark-torch-ptq instead.
---

Read and follow the instructions in `.claude/skills-impl/l2-workflows/onnx/quark-onnx-ptq-workflow/SKILL.md`.
