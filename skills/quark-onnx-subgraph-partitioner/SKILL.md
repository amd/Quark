---
name: quark-onnx-subgraph-partitioner
description: >
  Partition an ONNX model graph into named functional subgraphs and emit a
  subgraph_partition.json file. Use when the user wants to understand a model's
  high-level structure, document its architectural blocks, or prepare a partition
  for downstream workflows such as mixed-precision quantization, layer-wise
  profiling, or partial deployment. Triggers on "partition my ONNX model",
  "generate a subgraph JSON", "which nodes belong to the attention blocks",
  "show me the model architecture", "split the model into blocks", "group
  layers for AMP", or any request that needs a named block decomposition of an
  ONNX graph. Works on both float and quantized ONNX models: for quantized
  models, automatically detects Q/DQ wrappers and anchors subgraph boundaries
  on compute nodes rather than quantization wrapper nodes. Handles any ONNX
  architecture: CNNs, Transformers, multi-modal models, detection models with
  FPN/PAN necks, and BEV perception models.
---

Read and follow [the bundled implementation](../_legacy_impl/l1-atomic/onnx/quark-onnx-subgraph-partitioner/SKILL.md).
