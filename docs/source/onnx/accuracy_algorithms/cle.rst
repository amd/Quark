.. Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

Quantizing Using CrossLayerEqualization (CLE)
=============================================

CrossLayerEqualization (CLE) can equalize the weights of consecutive convolution layers, making the model weights easier to perform per-tensor quantization. For more details, please refer the paper `link <https://arxiv.org/abs/1906.04721>`__. Experiments show that the CLE technique improves PTQ accuracy for many models, especially those with depthwise convolutional layers, such as MobileNet and ShuffleNet.

Here is a simple example showing how to apply the CLE algorithm on an A8W8 (Activation-8bit-Weight-8bit) quantization.

.. code-block:: python

    from quark.onnx import ModelQuantizer
    from quark.onnx.quantization.config import QConfig
    from quark.onnx.quantization.config.spec import QLayerConfig, UInt8Spec, Int8Spec
    from quark.onnx.quantization.config.algorithm import CLEConfig

    quant_config = QLayerConfig(activation=UInt8Spec(), weight=Int8Spec())

    cle_config = CLEConfig(cle_steps=1, cle_scale_append_bias=True)

    config = QConfig(
        global_config=quant_config,
        algo_config=[cle_config],
    )

    quantizer = ModelQuantizer(config)
    quantizer.quantize_model(input_model_path, quantized_model_path, calib_data_reader)

Arguments
---------

Here we only list a few important and commonly used arguments, please refer to the documentation of full arguments list for more details.

  - **cle_steps**: (Int) Specifies the steps for CrossLayerEqualization execution when include_cle is set to true. The default is 1. When set to -1, adaptive CrossLayerEqualization steps are conducted. The default value is 1.

  - **cle_scale_append_bias**: (Boolean) Whether the bias is included when calculating the scale of the weights. The default value is True.

Example
=======

.. note::

   For information on accessing AMD Quark ONNX examples, refer to :doc:`Accessing ONNX Examples <../onnx_examples>`.
   This example and the relevant files are available at ``/onnx/accuracy_improvement/cle``

This :doc:`example <../example_quark_onnx_cle>` demonstrates quantizing a resnet152 model using the AMD Quark ONNX quantizer.
