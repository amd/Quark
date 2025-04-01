Quark for Ryzen AI NPU
======================

Here you will find references on how you can leverage Quark to seamlessly run quantized models on the Ryzen AI NPU.
Ryzen AI leverages ONNX models to represent models and execute them through ONNX Runtime.

To help you get started, we also have examples at the :ref:`ONNX Examples <ryzenai_onnx_examples>` page!

.. toctree::
   :caption: Resources
   :maxdepth: 1

   Quick Start for Ryzen AI <tutorial_quick_start_for_ryzenai.rst>
   Best Practice for Ryzen AI in AMD Quark ONNX <ryzen_ai_best_practice.rst>
   Auto-Search for Ryzen AI ONNX Model Quantization <../../onnx/example_quark_onnx_ryzenai>
   Quantizing LLMs for ONNX Runtime GenAI <tutorial_uint4_oga>
   FP32/FP16 to BF16 Model Conversion <tutorial_convert_fp32_or_fp16_to_bf16.rst>
   Power-of-Two Scales (Xint8) Quantization <tutorial_xint8_quantize.rst>
   Float Scales (A8W8 and A16W8) Quantization <tutorial_a8w8_and_a16w8_quantize.rst>

Quark also delivers a plethora of post-processing tools that might be of use for Ryzen AI. refer to the :doc:`ONNX Tools <../../onnx/tools>` to learn more!
