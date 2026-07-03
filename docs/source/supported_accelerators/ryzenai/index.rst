.. Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

Quark for Ryzen AI NPU
======================

This section provides guidance on leveraging AMD Quark to deploy quantized models on the Ryzen AI Neural Processing Unit (NPU).
By utilizing the `Ryzen AI Software <https://ryzenai.docs.amd.com/en/latest/index.html>`_, developers can seamlessly run optimized models trained in PyTorch or TensorFlow on Ryzen AI-enabled processors.
Ryzen AI Software has integrated tools and libraries like AMD Quark, ONNX Runtime and the Vitis AI Execution Providers (EP) that facilitates efficient inference across various accelerators, including CPU and integrated GPU (iGPU), in addition to the NPU.

.. image:: ../../_static/quark_ryzenai_overview.png
   :alt: Quark Ryzen AI Overview
   :align: center
   :width: 80%

Development Flow Steps
----------------------

- Trained Models: Trained models in popular frameworks such as PyTorch / TensorFlow are exported to ONNX format, to leverage ONNX Runtime to run on Ryzen AI supported processors.

- Model Quantization: Use AMD Quark quantizer tools to convert your ONNX model into a quantized version, using the following quantization schemes:
  - For CNN models: INT8 or BF16
  - For Transformer models: BF16
  - For LLMs: INT4 or BF16

- Deployment and Inference: Deploy the quantized on Ryzen AI-enabled hardware through ONNX runtime and Vitis AI Execution Provider.

AMD Quark provides advanced tools for model quantization. This documentation will help you navigate the capabilities of Quark to run with Ryzen AI.

Here you will find references on how you can leverage Quark to seamlessly run quantized models on the Ryzen AI NPU.
Ryzen AI leverages ONNX models to represent models and execute them through ONNX Runtime.

.. toctree::
   :hidden:
   :caption: Resources
   :maxdepth: 0

   Quick Start for Ryzen AI <tutorial_quick_start_for_ryzenai.rst>
   Best Practice for Ryzen AI <ryzen_ai_best_practice.rst>
   INT8/INT16 Quantizations <int8_int16_quantization.rst>
   FP32/FP16 to BF16 Model Conversion <tutorial_convert_fp32_or_fp16_to_bf16.rst>
   Quantize LLMs for ONNX Runtime GenAI <tutorial_uint4_oga>
   AutoSearch for Ryzen AI <../../tutorials/onnx/ryzen_ai/auto_search_for_ryzen_ai/auto_search_resnet50/onnx_ryzen_ai_auto_search_resnet50_tutorial>
   Efficiency Improvement <../../onnx/efficiency_improvement.rst>
   Troubleshooting <../../onnx/onnx_troubleshooting.rst>

Quark also delivers a plethora of post-processing tools that might be of use for Ryzen AI. refer to the :doc:`ONNX Tools <../../onnx/tools>` to learn more!
