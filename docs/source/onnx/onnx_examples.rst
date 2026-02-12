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

   Block Floating Point (BFP) <../tutorials/onnx/accuracy_improvement/bfp/onnx_bfp_tutorial>
   MX Formats <../tutorials/onnx/accuracy_improvement/MX/onnx_MX_tutorial>
   Fast Finetune AdaRound <../tutorials/onnx/accuracy_improvement/adaround/onnx_adaround_tutorial>
   Fast Finetune AdaQuant <../tutorials/onnx/accuracy_improvement/adaquant/onnx_adaquant_tutorial>
   Cross-Layer Equalization (CLE) <../tutorials/onnx/accuracy_improvement/cle/onnx_cle_tutorial>
   Layer-wise Percentile <../tutorials/onnx/accuracy_improvement/layerwise/onnx_layerwise_tutorial>
   GPTQ <../tutorials/onnx/accuracy_improvement/gptq/onnx_gptq_tutorial>
   Mixed Precision <../tutorials/onnx/accuracy_improvement/mixed_precision/onnx_mixed_precision_tutorial>
   Smooth Quant <../tutorials/onnx/accuracy_improvement/smooth_quant/onnx_smooth_quant_tutorial>
   QuaRot <example_quark_onnx_quarot>
   Auto-Search for Ryzen AI Yolov8 ONNX Model Quantization <../tutorials/onnx/ryzen_ai/auto_search_for_ryzen_ai/auto_search_yolov8/onnx_ryzen_ai_auto_search_yolov8_tutorial>
   Auto-Search for Ryzen AI MobileNetv2-50 ONNX Quantization with Custom Evaluator <../tutorials/onnx/ryzen_ai/auto_search_for_ryzen_ai/auto_search_mobilenetv2_50_custom_evaluator/onnx_ryzen_ai_auto_search_mobilenetv2_50_tutorial>
   Auto-Search for Ryzen AI Resnet50 ONNX Model Quantization <../tutorials/onnx/ryzen_ai/auto_search_for_ryzen_ai/auto_search_resnet50/onnx_ryzen_ai_auto_search_resnet50_tutorial>

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

.. toctree::
   :caption: Crypto Mode
   :maxdepth: 1

   Quantizing a ResNet50 model in crypto mode <../tutorials/onnx/crypto_mode/onnx_crypto_mode_tutorial>

.. _ryzenai_onnx_examples:
.. toctree::
   :caption: Ryzen AI Quantization
   :maxdepth: 1

   Best Practice for Quantizing an Image Classification Model <../tutorials/onnx/ryzen_ai/resnet50/onnx_ryzen_ai_resnet50_tutorial>
   Best Practice for Quantizing an Object Detection Model  <../tutorials/onnx/ryzen_ai/yolov8/onnx_ryzen_ai_yolov8_tutorial>
