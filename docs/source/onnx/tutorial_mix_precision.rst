.. Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

Mixed Precision
===============

.. note::

    In this documentation, **AMD Quark** is sometimes referred to simply as **"Quark"** for ease of reference. When you  encounter the term "Quark" without the "AMD" prefix, it specifically refers to the AMD Quark quantizer unless otherwise stated. Please do not confuse it with other products or technologies that share the name "Quark."

As the scale and complexity of AI models continue to grow, optimizing their performance and efficiency becomes a top priority. Quantizing models to mixed precision emerges as a powerful technique, allowing AI practitioners to balance computational speed, memory usage, and model accuracy. This tutorial introduces the characteristics and usage of AMD Quark for ONNX's mixed precision.

What is Mixed Precision Quantization?
-------------------------------------

Mixed precision quantization involves using different precision levels for different parts of a neural network, such as using 8-bit integers for some layers while retaining higher precision, for example, 16-bit or 32-bit floating point, for others. This approach leverages the fact that not all parts of a model are equally sensitive to quantization. By carefully selecting which parts of the model can tolerate lower precision, you achieve significant computational savings while minimizing the impact on model accuracy.

Benefits of Mixed Precision Quantization
----------------------------------------

1. **Enhanced Efficiency**: By using lower precision where possible, mixed precision quantization significantly reduces computational load and memory usage, leading to faster inference times and lower power consumption.

2. **Maintained Accuracy**: By selectively applying higher precision to sensitive parts of the model, mixed precision quantization minimizes the accuracy loss that typically accompanies uniform quantization.

3. **Flexibility**: Mixed precision quantization is adaptable to various types of neural networks and can be tailored to specific hardware capabilities, making it suitable for a wide range of applications.

Mixed Precision Quantization in AMD Quark for ONNX
--------------------------------------------------

AMD Quark for ONNX is designed to push the boundaries of what is possible with mixed precision. Here is what sets it apart:

1. **Support for All Types of Granularity**

Granularity refers to the level at which precision can be controlled within a model. AMD Quark for ONNX mixed precision supports:

- **Element-wise Granularity**

Element-wise mixed precision allows assigning different numeric precision levels to activations, weight and bias. For example, assign INT16 to activation to preserve dynamic range, INT8 to weight for efficient storage and computation and INT32 to bias for precision and overflow safety.

- **OpType-wise Granularity**

In practical use, it is sometimes possible to specify a certain precision for a specific operator type, in which case several layers of the same operator type will use the same precision. For example, in an INT8 quantized model, specifying all 'Softmax' layers as INT16.

- **Layer-wise Granularity**

Different layers of a neural network can have varying levels of sensitivity to quantization. Layer-wise mixed precision assigns precision levels to layers based on their sensitivity, optimizing both performance and accuracy. For example, INT16 to sensitive layers for high accuracy while INT8 to others for efficient inference.

- **Tensor-wise Granularity**

Tensor-wise mixed precision enables assigning different precision levels to individual tensors within a layer. For example, in an INT8 quantized model, specifying a convolution layer's input as INT16.

2. **Support for Various Data Types**

AMD Quark for ONNX mixed precision is not limited to a few integer data types, it supports a wide range of precision levels, including but not limited to:

- **More Integer Data Types**

Traditional INT8/UINT8 for significant memory and computation savings, INT16/UINT16 for higher precision and INT32/UINT32 for experimental usage.

- **Half Floating-Point Data Types**

Float16 and BFloat16, the former can be used for iGPU/GPU applications, while the latter can be used for NPU deployment.

- **Block Floating-Point Data Types**

The bit-width for shared exponents and elements can be set arbitrarily. The typical data type is BFP16.

- **Microexponents Data Types**

Supports all the Microexponents data types, including MX4, MX6 and MX9.

- **Microscaling Data Types**

Supports all the Microscaling data types, including MXINT8, MXFP8_E4M3, MXFP8_E5M2, MXFP6_E3M2, MXFP6_E2M3 and MXFP4.

How to Enable Mixed Precision in AMD Quark for ONNX?
----------------------------------------------------

Here, Int8 mixed with Int16 is used as an example to illustrate how to build configurations for mixed precision quantization.
In fact, you can mix any two other data types equally.

- **Element-wise**

In this configuration, Int16 is assigned to activations and Int8 to weights.

.. code-block:: python

   from quark.onnx import QConfig, QLayerConfig, Int16Spec, Int8Spec, ModelQuantizer

   # Build the configuration
   global_config = QLayerConfig(input_tensors=Int16Spec(), weight=Int8Spec())
   quant_config = QConfig(global_config=global_config)

   # Create an ONNX quantizer
   quantizer = ModelQuantizer(quant_config)

   # Quantize the ONNX model. Users need to provide the input model path, output model path,
   # and a data reader for calibration.
   quantizer.quantize_model(input_model_path, output_model_path, data_reader)


- **OpType-wise**

In this configuration, Int16 is assigned to the 'Softmax' operator.

.. code-block:: python

   from quark.onnx import QConfig, QLayerConfig, Int16Spec, Int8Spec, ModelQuantizer

   global_config = QLayerConfig(input_tensors=Int8Spec(), weight=Int8Spec())
   layer_type_config = {QLayerConfig(input_tensors=Int16Spec(), weight=Int16Spec(), biast=Int16Spec(), output_tensors=Int16Spec()): ['Softmax']}
   quant_config = QConfig(global_config=global_config, layer_type_config=layer_type_config)


- **Layer-wise**

This is one of the common configurations for deploying models on hardware devices, where the sensitive layers are quantized into Int16 to maintain accuracy, and the remaining layers are quantized into Int8.

.. code-block:: python

   from quark.onnx import QConfig, QLayerConfig, Int16Spec, Int8Spec, ModelQuantizer

   global_config = QLayerConfig(input_tensors=Int8Spec(), weight=Int8Spec())
   specific_layer_config = {QLayerConfig(input_tensors=Int16Spec(), weight=Int16Spec(), biast=Int16Spec(), output_tensors=Int16Spec()): ['/model/conv_1', '/model/gemm_1']}
   quant_config = QConfig(global_config=global_config, specific_layer_config=specific_layer_config)


- **Tensor-wise**

Certain tensors in a neural network are particularly sensitive to quantization, including activation, weight and even bias tensors. Applying appropriate precision for these sensitive tensors can help maintain model accuracy while reaping the benefits of quantization. Therefore, after identifying these tensors through sensitivity analysis, you can set the precision separately for them.

.. code-block:: python

   from quark.onnx import QConfig, QLayerConfig, Int16Spec, Int8Spec, ModelQuantizer, QuantType

   global_config = QLayerConfig(input_tensors=Int8Spec(), weight=Int8Spec())
   extra_options = { 'TensorQuantOverrides': { '/model/conv_1/output_0': [ { 'quant_type': QuantType.Int16 } ], }, }
   quant_config = QConfig(global_config=global_config, extra_options=extra_options)

It is worth mentioning that, aside from the basic element-wise configuration, other mixed precision configurations can be combined in any manner. For example, you can use combinations like "Element-wise + OpType-wise + Layer-wise" or "Element-wise + OpType-wise + Layer-wise + Tensor-wise". However, if a single tensor is assigned different levels of precision in the combined configuration, the setting of the smaller granularity will override the one of larger granularity.

Automatic Mixed Precision based on Sensitivity Analysis
--------------------------------------------------------

The previous examples are manually specified mixed precision, but in the practical applications automatically identifying sensitive layers and then applying mixed precision becomes more critical.

AMD Quark for ONNX supports automatic mixed precision as follows:

**Step 1** Sensitivity analysis. This step can involve profiling the model with a new precision settings and measuring the impact on accuracy.

**Step 2** Sort layers by sensitivity. Layers that show significant accuracy degradation when quantized are deemed "sensitive" and are kept at higher precision. Less sensitive parts can be quantized more aggressively to lower precision without significant impact on overall model performance.

**Step 3** Perform mixed precision operations. Perform layer by layer until reach the accuracy target which is specified by users.

We provide two types of accuracy target: general L2 Norm metric and Top1 metric specific to image classification models. Here is a simple example of how to use the L2 Norm metric to achieve automatic mixed precision:

.. code-block:: python

   from quark.onnx import QConfig, QLayerConfig, Int8Spec, Int16Spec, ModelQuantizer, AutoMixprecisionConfig

   auto_mixprecision_algo = AutoMixprecisionConfig(target_op_type=["Conv", "ConvTranspose", "Gemm", "MatMul"],
                                                   act_target_quant_type=Int8,
                                                   weight_target_quant_type=Int16,
                                                   output_index=0,
                                                   l2_target=0.1)

   # Build the configuration
   quant_config = QConfig(global_config=QLayerConfig(input_tensors=Int16Spec(), weight=Int8Spec()),
                          algo_config=[auto_mixprecision_algo])

   # Create an ONNX quantizer
   quantizer = ModelQuantizer(quant_config)

   # Quantize the ONNX model. Users need to provide the input model path, output model path,
   # and a data reader for calibration.
   quantizer.quantize_model(input_model_path, output_model_path, data_reader)

For a detailed example of using Top1 metric for mixed precision, refer to the :doc:`Mixed Precision Example <../tutorials/onnx/accuracy_improvement/mixed_precision/onnx_mixed_precision_tutorial>`.
