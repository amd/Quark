.. Copyright (C) 2025 - 2026, Advanced Micro Devices, Inc. All rights reserved.

File-to-File LLM Quantization
=============================

Challenges in Quantizing Ultra-Large Models
-------------------------------------------

For quantization schemes that operate independently on each tensor — such as **weight-only quantization** (FP8, MXFP4, INT4, etc.) or **dynamic activation quantization + weight quantization** — standard tools still require loading the **entire model** into GPU memory. For a 600B+ parameter model this means hundreds of gigabytes, leading to **Out of Memory (OOM)** failures even on multi-GPU nodes.

This is wasteful: these quantization schemes are **per-tensor operations** that do not require the full model graph. Each tensor can be quantized independently.

The Solution: File-to-File Quantization, No Full Model Loading
--------------------------------------------------------------

**File-to-File Quantization** exploits this independence by reading each ``.safetensors`` file, quantizing the weights it contains, and writing the result directly to a new file — without ever loading the full model into memory.

.. list-table:: Standard vs. File-to-File quantization
   :widths: 28 36 36
   :header-rows: 1

   * - Item
     - Standard quantization
     - File-to-file quantization
   * - What gets loaded
     - The **entire model** (all files / full state dict)
     - **One file at a time** (plus optional recovery metadata)
   * - Core operation
     - Per-tensor weight quantization
     - Per-tensor weight quantization (**same math**)
   * - Peak memory driver
     - Full model size (often **100s of GB** for 600B+)
     - Largest single file (typically **~5–10 GB**)
   * - Pre-quantized input handling
     - Typically assumes BF16/FP16 inputs
     - Can **recover then re-quantize** (e.g., FP8 or compressed-tensors)
   * - Output
     - Quantized weights + exported artifacts
     - Quantized weights + exported artifacts

Both approaches produce the **same per-tensor quantization** results. File-to-file mode supports **weight-only quantization** (e.g., FP8, MXFP4, INT4) and **dynamic activation quantization + weight quantization**, where each tensor can be processed independently — no calibration data, forward pass, or full model graph is required. The only difference is how tensors are loaded: the standard approach loads the entire model at once, while file-to-file reads and writes one ``.safetensors`` file at a time.

Supported Input Formats
-----------------------

.. list-table::
   :widths: 15 25 35 25
   :header-rows: 1

   * - Format
     - Quant Method
     - Description
     - Dependencies
   * - **BF16 / FP16**
     - —
     - Standard HuggingFace model, loaded directly
     - —
   * - **FP8**
     - ``fp8``
     - FP8 weights with ``_scale_inv`` tensors; dequantized before re-quantization
     - **Triton**
   * - **compressed-tensors**
     - ``compressed-tensors``
     - HuggingFace-style packed weights; decompressed before re-quantization
     - **compressed-tensors**

Usage Examples
--------------

Example 1: File-to-File Quantization via Python API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This approach mirrors the ``_build_quant_config`` helper in ``quantize_quark.py`` — it uses ``LLMTemplate`` to auto-detect the model architecture and build the quantization config from a scheme name, avoiding manual ``QConfig`` construction.

.. code-block:: python

   import json
   from quark.torch import LLMTemplate, ModelQuantizer

   model_path = "/path/to/model"
   save_path = "/path/to/output"

   # Read model type from config.json
   with open(f"{model_path}/config.json") as f:
       model_type = json.load(f)["model_type"]

   # Build quant config
   template = LLMTemplate.get(model_type)
   quant_config = template.get_config(
       scheme="mxfp4",                              # or "fp8", "int4_wo_128", etc.
       exclude_layers=["*self_attn*", "*lm_head"],
   )

   # Run file-to-file quantization
   quantizer = ModelQuantizer(quant_config)
   quantizer.direct_quantize_checkpoint(
       pretrained_model_path=model_path,
       save_path=save_path,
   )

Example 2: File-to-File Mode via ``quantize_quark.py``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``quantize_quark.py`` script supports a ``--file2file_quantization`` flag that bypasses model loading and the standard quantization pipeline. It reads ``config.json`` to determine the model type, builds ``quant_config`` using the same ``LLMTemplate`` / ``--quant_scheme`` mechanism, and directly quantizes safetensors files in a file-to-file manner.

.. code-block:: bash

   python examples/torch/language_modeling/llm_ptq/quantize_quark.py \
       --model_dir /path/to/model \
       --quant_scheme mxfp4 \
       --exclude_layers "*self_attn*" "*mlp.gate" "*mlp.gate.linear" "*lm_head" \
       --output_dir /path/to/output \
       --file2file_quantization \
       --skip_evaluation


Requirements
------------

- **PyTorch**
- **safetensors**
- **Triton** (optional, only needed for FP8 input format dequantization)
- **compressed-tensors** (optional, only needed for compressed-tensors input format)
