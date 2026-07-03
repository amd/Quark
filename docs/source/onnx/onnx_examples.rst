.. Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

Accessing ONNX Examples
=======================

Users can get the example code after downloading and unzipping ``amd_quark.zip`` (referring to :doc:`Installation Guide <../install>`).
The example folder is in amd_quark.zip.

   Directory Structure of the ZIP File:

   ::

         + amd_quark.zip
            + amd_quark.whl
            + examples    # HERE IS THE EXAMPLES
               + torch
                  + language_modeling
                  + diffusers
                  + ...
               + onnx # HERE ARE THE ONNX EXAMPLES
                  + image_classification
                  + object_detection
                  + ...
            + ...

ONNX Examples in AMD Quark for This Release
-------------------------------------------

.. toctree::
   :hidden:
   :caption: Improving Model Accuracy
   :maxdepth: 1

   QuaRot <example_quark_onnx_quarot>

.. toctree::
   :hidden:
   :caption: Dynamic Quantization
   :maxdepth: 1

   Quantizing an Llama-2-7b Model <example_quark_onnx_dynamic_quantization_llama2>
   Quantizing an OPT-125M Model <example_quark_onnx_dynamic_quantization_opt>

.. toctree::
   :hidden:
   :caption: Language Models
   :maxdepth: 1

   Quantizing an OPT-125M Model <example_quark_onnx_language_models>

.. toctree::
   :hidden:
   :caption: Weights-Only Quantization
   :maxdepth: 1

   Quantizing an Llama-2-7b Model Using the ONNX MatMulNBits <example_quark_onnx_weights_only_quant_int4_matmul_nbits_llama2>
   Quantizing Llama-2-7b model using MatMulNBits <example_quark_onnx_weights_only_quant_int8_qdq_llama2>
