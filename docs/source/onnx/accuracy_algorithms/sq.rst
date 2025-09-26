.. Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

SmoothQuant (SQ)
================

SmoothQuant (SQ) is another technique used to improve PTQ accuracy. It smooths the outliers of the activation so that it loses as little precision as possible during quantization. Experiments show that using the SQ technique can improve the PTQ accuracy of some models, especially for models with a large number of outliers in the activation.

Here is a simple example showing how to apply the QuaRot algorithm on an A8W8 (Activation-8bit-Weight-8bit) quantization.

.. code-block:: python

    from quark.onnx import ModelQuantizer
    from quark.onnx.quantization.config import QConfig
    from quark.onnx.quantization.config.spec import QLayerConfig, UInt8Spec, Int8Spec
    from quark.onnx.quantization.config.algorithm import SmoothQuantConfig

    quant_config = QLayerConfig(activation=UInt8Spec(), weight=Int8Spec())

    sq_config = SmoothQuantConfig(alpha=0.5)

    config = QConfig(
        global_config=quant_config,
        algo_config=[sq_config],
        OpTypesToQuantize=['MatMul', 'Gemm'],
    )

    quantizer = ModelQuantizer(config)
    quantizer.quantize_model(input_model_path, quantized_model_path, calib_data_reader)

Arguments
---------

Here we only list a few important and commonly used arguments, please refer to the documentation of full arguments list for more details.

  - **alpha**: (Float) This parameter controls how much difficulty we want to migrate from activation to weights. The default value is 0.5.

Example
-------

.. note::

   For information on accessing AMD Quark ONNX examples, refer to :doc:`Accessing ONNX Examples <../onnx_examples>`.
   This example and the relevant files are available at ``/onnx/accuracy_improvement/smooth_quant``

This :doc:`example <../example_quark_onnx_smoothquant>` demonstrates quantizing an opt-125m model using the AMD Quark ONNX quantizer. It also shows how to use the Smooth Quant algorithm.
