.. Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

Mix Precision Auto-Search
=========================

The Mix Precision Auto-Search workflow automatically finds the optimal
mixed-precision quantization configuration for large language models running
on AMD Instinct GPUs via vLLM.

Given an accuracy loss budget—for example, GSM8K must not drop by more than
2%—it uses a hardware-aware decode Roofline to walk adjacent performance
candidates and selects the fastest evaluated configuration whose accuracy
stays within the threshold.

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
2. **Generate candidate configs** — constrained by the target hardware and
   requested mode subset. With ``--file2file_quantization``, candidates containing
   calibration-dependent ``fp8`` or ``mxfp4_fp8`` (W4A8) modes are reported in a
   warning and removed before evaluation.
3. **Compute the Roofline** — walk the meta model's Linear shapes and score
   candidates with per-op GEMM, FusedMoE, and SDPA compute/memory ceilings on
   the target GPU, with aggregate memory as a fallback. The default workload is
   ``ISL=8192`` / ``OSL=1024``; a separate prefill Roofline is reported but does
   not affect ordering.
4. **Evaluate baseline** — run GSM8K on the original (unquantized) vLLM model.
5. **Search loop** — start from the hardware anchor, walk adjacent Roofline
   candidates, re-quantize, evaluate, check the threshold, and reset.
6. **Select best config** — the highest-scoring evaluated config under the
   Roofline model that passes.
7. **Export** (optional) — apply best config and save as safetensors. With
   ``--file2file_quantization``, the best calibration-free config is exported one
   checkpoint shard at a time without loading the full model. Hugging Face model
   IDs are resolved to the local cache automatically; local inputs must contain
   ``config.json`` and at least one ``.safetensors`` shard.

MI300 and MI325 use ``mlp=ptpc_fp8`` as the preferred anchor. MI355 uses
``mlp=mxfp4``. All other layer partitions start as ``native``.

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
       --export_best_model \
       --file2file_quantization

Further Reading
---------------

* `Example script and README <https://gitenterprise.xilinx.com/AMDNeuralOpt/Quark/blob/main/examples/torch/experimental/mix_precision/README.md>`_
* `API design document <https://gitenterprise.xilinx.com/AMDNeuralOpt/Quark/blob/main/quark/experimental/torch/mix_precision/README.md>`_
