.. Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

Exporting Quantized Models
==========================

Quark torch not only supports our own torch export format `Quark format` (Json-Pth),
but also support exporting in popular formats requested by downstream tools, including `ONNX`, `format for Hugging Face & vLLM (HF format)`, and `GGUF`.

For diffusion models, ``quark.torch.export_safetensors`` detects a
HuggingFace Diffusers ``ModelMixin`` and writes a checkpoint that reloads
directly through ``DiffusionPipeline.from_pretrained`` /
``ModelMixin.from_pretrained``.  See
:doc:`Using Quark-Quantized Diffusion Models with HuggingFace Diffusers <../example_quark_torch_huggingface_diffusers>`.

.. toctree::
   :hidden:
   :caption: Exporting Quantized Models
   :maxdepth: 1

   ONNX format <quark_export_onnx>
   Hugging Face format (safetensors) <quark_export_hf>
   GGUF format <quark_export_gguf>
