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


API Reference: ``direct_quantize_checkpoint``
---------------------------------------------

.. code-block:: python

   ModelQuantizer.direct_quantize_checkpoint(
       pretrained_model_path,
       save_path,
       keep_excluded_layers_as_original_model_state=False,
       weight_converters=None,
       device=None,
       presharded_weights=None,
   )

.. list-table::
   :widths: 25 12 63
   :header-rows: 1

   * - Parameter
     - Default
     - Description
   * - ``pretrained_model_path``
     - *(required)*
     - Path to the pretrained model directory containing ``.safetensors`` files.
   * - ``save_path``
     - *(required)*
     - Directory path to save the quantized ``.safetensors`` files and all auxiliary files (``config.json``, ``model.safetensors.index.json``, tokenizer files, etc.).
   * - ``keep_excluded_layers_as_original_model_state``
     - ``False``
     - If ``True``, excluded layers that are already quantized in the source checkpoint keep their original model-state format in the output. Useful when the source is a pre-quantized checkpoint and certain layers should pass through unchanged.
   * - ``weight_converters``
     - ``None``
     - Optional list of :py:class:`WeightConverter` instances to transform tensors after precision recovery and before quantization. See `Weight Transformation with WeightConverter`_ below.
   * - ``device``
     - ``None``
     - Device for tensor operations (e.g. ``"cuda"``, ``"cuda:0"``, ``"cpu"``). Defaults to ``"cuda"`` when ``None``.
   * - ``presharded_weights``
     - ``None``
     - Optional pre-loaded shard dictionary. When provided, the method uses these tensors directly instead of reading from ``pretrained_model_path``.

Weight Transformation with WeightConverter
------------------------------------------

:py:class:`WeightConverter` allows post-quantization weight transformations on each ``.safetensors`` shard as it passes through the file-to-file pipeline. Converters run **after** precision recovery (dequantization of pre-quantized inputs) and **before** Quark quantization.

**Use cases:**

- Splitting a fused projection (e.g. ``gate_up_proj``) into separate tensors (``gate_proj``, ``up_proj``) that Quark quantizes individually.
- Renaming tensor keys to match a different model variant's naming convention.
- Applying custom packing or layout transformations after quantization.

Each converter receives the tensor dictionary for one ``.safetensors`` shard and returns a modified dictionary. Converters in the list are applied in order.

**Example — split fused gate/up projection:**

.. code-block:: python

   from quark.torch import LLMTemplate, ModelQuantizer
   from quark.torch.quantization.weight_convert import Chunk, WeightConverter

   template = LLMTemplate.get("qwen")
   quant_config = template.get_config(scheme="mxfp4")

   weight_converters = [
       WeightConverter(
           "gate_up_proj.weight",
           ["gate_proj.weight", "up_proj.weight"],
           operations=[Chunk(dim=0)],
       ),
   ]

   quantizer = ModelQuantizer(quant_config)
   quantizer.direct_quantize_checkpoint(
       pretrained_model_path="/path/to/model",
       save_path="/path/to/output",
       weight_converters=weight_converters,
   )

.. note::

   ``WeightConverter`` is suited for **post-recovery single-suffix rename** or **one-source split** operations. It is not suited for multi-source merges, cross-shard scale pairing, or dtype conversions such as FP4 → FP8. For those cases, generate a normalization script that pre-processes the checkpoint before running file-to-file quantization.

Progressive (Two-Step) Quantization
------------------------------------

Progressive quantization applies quantization in **two sequential stages** within a single file-to-file pass:

1. The weight is first quantized to an intermediate format (e.g. FP8 E4M3 per-tensor).
2. The intermediate result is then re-quantized to the final target format (e.g. INT4 per-channel).

This produces better accuracy than direct single-step quantization for aggressive targets such as INT4, because the first stage preserves more information than the final format alone.

**Built-in progressive scheme — ``int4_fp8``:**

The ``int4_fp8`` scheme (W4A8) uses progressive quantization internally:

- **Weight stage 1**: FP8 E4M3 per-tensor static quantization.
- **Weight stage 2**: INT4 per-channel static quantization of the FP8 result.
- **Activation**: FP8 E4M3 per-tensor dynamic quantization.

This matches the AMD-Quark W4A8 recipe used by models such as ``amd/Kimi-K2-Thinking-W4A8``.

.. code-block:: python

   import json
   from quark.torch import LLMTemplate, ModelQuantizer

   model_path = "/path/to/model"
   save_path = "/path/to/output"

   with open(f"{model_path}/config.json") as f:
       model_type = json.load(f)["model_type"]

   template = LLMTemplate.get(model_type)
   quant_config = template.get_config(scheme="int4_fp8")   # progressive W4A8

   quantizer = ModelQuantizer(quant_config)
   quantizer.direct_quantize_checkpoint(
       pretrained_model_path=model_path,
       save_path=save_path,
   )

.. code-block:: bash

   python examples/torch/language_modeling/llm_ptq/quantize_quark.py \
       --model_dir /path/to/model \
       --quant_scheme int4_fp8 \
       --file2file_quantization \
       --output_dir /path/to/output \
       --skip_evaluation

**Custom progressive scheme via ``ProgressiveSpec``:**

.. code-block:: python

   from quark.torch.quantization.config.config import (
       QConfig, QLayerConfig, ProgressiveSpec,
       FP8E4M3PerTensorSpec, Int4PerChannelSpec,
   )
   from quark.torch import ModelQuantizer

   weight_spec = ProgressiveSpec(
       first_stage=FP8E4M3PerTensorSpec(
           observer_method="min_max", scale_type="float32", is_dynamic=False
       ),
       second_stage=Int4PerChannelSpec(
           symmetric=True, scale_type="float32",
           round_method="half_even", ch_axis=0, is_dynamic=False
       ),
   ).to_quantization_spec()

   quant_config = QConfig(global_quant_config=QLayerConfig(weight=weight_spec))
   quantizer = ModelQuantizer(quant_config)
   quantizer.direct_quantize_checkpoint(
       pretrained_model_path="/path/to/model",
       save_path="/path/to/output",
   )

**Output tensor naming for progressive quantization:**

Progressive quantization writes three tensors per weight:

.. list-table::
   :widths: 40 60
   :header-rows: 1

   * - Tensor key
     - Contents
   * - ``<layer>.weight``
     - Packed weight from stage 2 (e.g. INT4 values).
   * - ``<layer>.weight_scale``
     - Scale from stage 1 (e.g. FP8 per-tensor scale).
   * - ``<layer>.weight_scale_2``
     - Scale from stage 2 (e.g. INT4 per-channel scale).

Requirements
------------

- **PyTorch**
- **safetensors**
- **Triton** (optional, only needed for FP8 input format dequantization)
- **compressed-tensors** (optional, only needed for compressed-tensors input format)
