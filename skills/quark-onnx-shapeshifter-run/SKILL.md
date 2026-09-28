---
name: quark-onnx-shapeshifter-run
description: >
  Apply existing ShapeShifter graph passes to an .onnx model via the quark-cli shapeshifter CLI or
  a ShapeShifter YAML. Trigger for "run ShapeShifter on my .onnx", "apply an onnx_ pass", "fold
  batch norm / simplify / convert opset / fuse LayerNorm on my ONNX model", "preprocess my .onnx
  before quantization", "postprocess my quantized .onnx for XINT8/NPU". Operates on .onnx only. NOT
  authoring a new pass (use quark-create-shapeshifter-pass), NOT full quantization (use
  quark-onnx-ptq), NOT PyTorch models.
---

Read and follow [the bundled implementation](../_legacy_impl/l1-atomic/onnx/quark-onnx-shapeshifter-run/SKILL.md).
