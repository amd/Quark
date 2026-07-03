.. Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

Mix Precision Auto-Search
=========================

The Mix Precision Auto-Search workflow automatically finds the optimal
mixed-precision quantization configuration for large language models running
on AMD Instinct GPUs via vLLM.

Given an accuracy loss budget—for example, GSM8K must not drop by more than
2%—it searches from least to most aggressive quantization and selects the
most aggressively quantized configuration whose accuracy stays within the
threshold, without any manual parameter testing.

Supported Quantization Modes
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Hardware
     - Supported Modes
   * - MI300
     - ``native``, ``fp8``, ``ptpc_fp8``
   * - MI325
     - ``native``, ``fp8``, ``ptpc_fp8``
   * - MI355
     - ``native``, ``fp8``, ``ptpc_fp8``, ``mxfp4``, ``mxfp4_fp8``, ``mxfp6_e2m3``

Workflow Overview
-----------------

1. **Load model on meta device** — builds the layer graph for quantization
   config generation with no GPU memory cost.
2. **Generate candidate configs** — sorted from least to most aggressive
   quantization.
3. **Evaluate baseline** — run GSM8K on the original (unquantized) vLLM model.
4. **Search loop** — for each config in rank order, re-quantize, evaluate,
   check accuracy threshold, and reset.
5. **Select best config** — the most aggressively quantized config that passes
   the threshold.
6. **Export** (optional) — apply best config and save as safetensors.

Quick Start
-----------

.. note::

   This workflow requires the vLLM ROCm container. See the
   `example README <https://gitenterprise.xilinx.com/AMDNeuralOpt/Quark/blob/main/examples/torch/experimental/mix_precision/README.md>`_
   for full environment setup instructions.

.. code-block:: bash

   cd /workspace/quark/examples/torch/experimental/mix_precision

   # Search all modes for MI300
   python mix_precision.py \
       --model_dir /models/Qwen3.5-397B-A17B \
       --hardware mi300 \
       -tp 8 \
       --gpu-memory-utilization 0.8 \
       --export_best_model

Further Reading
---------------

* `Example script and README <https://gitenterprise.xilinx.com/AMDNeuralOpt/Quark/blob/main/examples/torch/experimental/mix_precision/README.md>`_
* `API design document <https://gitenterprise.xilinx.com/AMDNeuralOpt/Quark/blob/main/quark/experimental/torch/llm/mix_precision/README.md>`_
