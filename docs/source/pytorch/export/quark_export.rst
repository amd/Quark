.. Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

Exporting Quantized Models
==========================

Quark torch not only supports our own torch export format `Quark format` (Json-Pth),
but also support exporting in popular formats requested by downstream tools, including `ONNX`, `format for Hugging Face & vLLM (HF format)`, and `GGUF`.

.. toctree::
   :hidden:
   :caption: Exporting Quantized Models
   :maxdepth: 1

   ONNX format <quark_export_onnx>
   Hugging Face format (safetensors) <quark_export_hf>
   GGUF format <quark_export_gguf>
