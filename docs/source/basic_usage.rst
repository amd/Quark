.. Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

Getting started with AMD Quark
==============================

AMD Quark provides a streamlined approach to quantizing models in both PyTorch and ONNX formats, enabling efficient deployment across various hardware platforms.

Users need to choose which flow they will use for quantizing their model. Generally speaking, the PyTorch workflow is recommended for large language models (LLMs), otherwise the ONNX flow is recommended. Ryzen AI NPU is only supported by the ONNX flow, while PyTorch flow supports ROCm and CUDA accelerators.

Typically, quantizing a floating-point model with AMD Quark involves the following steps:

.. _basic-usage-quantization-steps:

1. Load the original floating-point model.
2. Define the data loader for calibration (optional).
3. Set the quantization configuration.
4. Use the AMD Quark API to perform an in-place replacement of the model's modules with quantized modules.
5. (Optional, only supported for PyTorch flow) Export the quantized model to other formats for deployment, such as ONNX, Hugging Face safetensors, etc.

.. The flowchart below will guide you in selecting the appropriate workflow based on your specific workload requirements.

.. TODO: Add a flowchart to help users to choose between Quark and ONNX flows
.. .. figure:: ./_static/quant/flowchart_quark_torch_or_onnx.png
..    :align: center
..
..    Caption for flowchart

Comparing Quark's ONNX and PyTorch capabilities
-----------------------------------------------

Each Quark workflow (PyTorch and ONNX) possesses its own set of features, data type support, and characteristics, catering to different model architectures and deployment scenarios.  Understanding these nuances is crucial for optimal quantization results.

+--------------------+-------------------------------------------------+-----------------------------------------------+
| Feature Name       | Quark for PyTorch                               | Quark for ONNX                                |
+====================+=================================================+===============================================+
| Data Type          | Float16 / Bfloat16 / Int3 / Int4 / Uint4 /      | Int8 / Uint8 / Int16 / Uint16 / Int32 /       |
|                    | Int8 / Uint8 / OCP_FP8_E4M3 / OCP_MXFP8_E4M3 /  | Uint32 / Float16 / Bfloat16                   |
|                    | OCP_MXFP6 / OCP_MXFP4 / OCP_MXINT8              |                                               |
+--------------------+-------------------------------------------------+-----------------------------------------------+
| Quant Mode         | Eager Mode / FX Graph Mode                      | ONNX Graph Mode                               |
+--------------------+-------------------------------------------------+-----------------------------------------------+
| Quant Strategy     | Static quant / Dynamic quant / Weight only      | Static quant / Weight only / Dynamic quant    |
+--------------------+-------------------------------------------------+-----------------------------------------------+
| Quant Scheme       | Per tensor / Per channel / Per group            | Per tensor / Per channel                      |
+--------------------+-------------------------------------------------+-----------------------------------------------+
| Symmetric          | Symmetric / Asymmetric                          | Symmetric / Asymmetric                        |
+--------------------+-------------------------------------------------+-----------------------------------------------+
| Calibration method | MinMax / Percentile / MSE                       | MinMax / Percentile / MinMSE /                |
|                    |                                                 | Entropy / NonOverflow                         |
+--------------------+-------------------------------------------------+-----------------------------------------------+
| Scale Type         | Float32 / Float16                               | Float32 / Float16                             |
+--------------------+-------------------------------------------------+-----------------------------------------------+
| KV-Cache Quant     | FP8 KV-Cache Quant                              | N/A                                           |
+--------------------+-------------------------------------------------+-----------------------------------------------+
| Supported Ops      | nn.Linear / nn.Conv2d / nn.ConvTranspose2d /    | Most ONNX ops.                                |
|                    | nn.Embedding / nn.EmbeddingBag                  | (:ref:`Full List <quark-onnx-supported-ops>`) |
+--------------------+-------------------------------------------------+-----------------------------------------------+
| Pre-Quant          | SmoothQuant                                     | QuaRot / SmoothQuant (Single_GPU/CPU) /       |
| Optimization       |                                                 | CLE / Bias Correction                         |
+--------------------+-------------------------------------------------+-----------------------------------------------+
| Quantization       | AWQ / GPTQ / Qronos                             | AdaQuant / AdaRound / GPTQ                    |
| Algorithm          |                                                 |                                               |
+--------------------+-------------------------------------------------+-----------------------------------------------+
| Export Format      | ONNX / Json-safetensors / GGUF(Q4_1)            | N/A                                           |
+--------------------+-------------------------------------------------+-----------------------------------------------+
| Operating Systems  | Linux (ROCm/CUDA) / Windows (CPU)               | Linux(ROCm/CUDA) / Windows(CPU)               |
+--------------------+-------------------------------------------------+-----------------------------------------------+

Next steps
----------

* :doc:`Getting started with Quark for ONNX models <onnx/basic_usage_onnx>`
* :doc:`Getting started with Quark for PyTorch models <pytorch/basic_usage_pytorch>`
