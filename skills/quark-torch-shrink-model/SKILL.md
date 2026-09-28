---
name: quark-torch-shrink-model
description: >
  Shrink a HuggingFace safetensors model to 1 hidden layer for fast debugging without loading
  the full model into memory. Use when the user wants to create a minimal model for debugging,
  reduce a large model to its smallest valid structure, generate a tiny model for testing Quark
  workflows, or validate layer detection and index rewriting without copying large tensors.
  Trigger for "shrink model", "minimal model for debug", "1-layer model", "tiny model for testing",
  "reduce model layers", "debug model structure", "fast model for testing", "shrink safetensors".
---

Read and follow [the bundled implementation](../_legacy_impl/l1-atomic/torch/quark-torch-shrink-model/SKILL.md).
