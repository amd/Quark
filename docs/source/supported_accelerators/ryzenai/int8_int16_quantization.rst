.. Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

INT8/INT16 Quantization for Ryzen AI
====================================

Introduction
------------

Ryzen AI NPU supports two major categories of quantization schemes, which differ primarily in how scaling factors are represented and applied during quantization:

1. **Power-of-Two Scales Quantization (XINT8)**
   This method uses power-of-two scaling factors with symmetric INT8 quantization for activations, weights, and biases. It is highly optimized for performance on Ryzen AI hardware, enabling efficient computation through bit-shift operations. However, due to its constrained scaling scheme, it may introduce accuracy degradation for certain models.

2. **Float Scales Quantization (A8W8 and A16W8)**
   These methods use floating-point scaling factors, providing greater flexibility and typically better accuracy.
   - **A8W8** uses INT8 activations and weights.
   - **A16W8** increases activation precision to INT16, further improving accuracy at the cost of performance.

Choosing between these approaches involves a trade-off between **performance and accuracy**:

- Use **XINT8** when maximum performance is required and minor accuracy loss is acceptable.
- Use **A8W8** for a balance between performance and accuracy.
- Use **A16W8** when higher accuracy is critical and additional compute cost is acceptable.

This documentation provides detailed guides for both quantization approaches, including:
- How to quantize float models
- How to evaluate quantization accuracy
- Techniques to improve quantized model performance

.. toctree::
   :hidden:
   :caption: Resources
   :maxdepth: 0

   Power-of-Two Scales (XINT8) Quantization <tutorial_xint8_quantize.rst>
   Float Scales (A8W8 and A16W8) Quantization <tutorial_a8w8_and_a16w8_quantize.rst>


Summary of Differences
----------------------

+----------------------+---------------------------+------------------------------+
| Feature              | XINT8                     | A8W8 / A16W8                 |
+======================+===========================+==============================+
| Scale Type           | Power-of-Two              | Floating-point               |
+----------------------+---------------------------+------------------------------+
| Activation Precision | INT8                      | INT8 (A8W8), INT16 (A16W8)   |
+----------------------+---------------------------+------------------------------+
| Weight Precision     | INT8                      | INT8                         |
+----------------------+---------------------------+------------------------------+
| Bias Precision       | INT32                     | INT32                        |
+----------------------+---------------------------+------------------------------+
| Quantization Scheme  | Symmetric                 | Symmetric                    |
+----------------------+---------------------------+------------------------------+
| Performance          | Highest                   | Medium (A8W8) / Lower (A16W8)|
+----------------------+---------------------------+------------------------------+
| Accuracy             | Lower (in some cases)     | Higher                       |
+----------------------+---------------------------+------------------------------+
| Hardware Efficiency  | Optimal (bit-shift ops)   | Less optimal                 |
+----------------------+---------------------------+------------------------------+


When to Use Each Method
-----------------------

- **Choose XINT8 if:**
  - You need maximum throughput on Ryzen AI NPU
  - Your model tolerates stricter quantization constraints
  - You prioritize latency and efficiency over accuracy

- **Choose A8W8 if:**
  - You want a balance between performance and accuracy
  - Your model shows noticeable degradation with XINT8

- **Choose A16W8 if:**
  - Accuracy is critical
  - Your model is sensitive to quantization errors
  - You can afford slightly lower performance


Next Steps
----------

To get started, refer to the detailed guides:

- :doc:`Power-of-Two Scales (XINT8) Quantization <tutorial_xint8_quantize>`
- :doc:`Float Scales (A8W8 and A16W8) Quantization <tutorial_a8w8_and_a16w8_quantize>`

Each guide includes:
- Step-by-step quantization workflow
- Example code snippets
- Accuracy evaluation methods
- Advanced techniques such as AdaRound and AdaQuant for improving results
