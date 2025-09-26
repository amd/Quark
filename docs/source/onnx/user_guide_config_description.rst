.. Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.

Configuring ONNX Quantization
=============================

Configuration of quantization in ``AMD Quark for ONNX`` is set by Python ``dataclass`` because it is rigorous and can help you avoid typos. We provide a class ``QConfig`` in ``quark.onnx.quantization.config.config`` for configuration, as demonstrated in the previous example. It can use ``get_default_config`` for predefined configurations.

The ``Config`` should be like:

.. code-block:: python

   from quark.onnx.quantization import QConfig
   quant_config = QConfig.get_default_config("xxx")

We define some default global configurations, including ``XINT8`` and ``A8W8``, which can be used like this:

.. code-block:: python

   quant_config = QConfig.get_default_config("A8W8")

More Quantization Default Configurations
----------------------------------------

AMD Quark for ONNX provides you with default configurations to quickly start model quantization.

-  ``INT8_CNN_DEFAULT``: Perform 8-bit, optimized for CNN quantization.
-  ``INT16_CNN_DEFAULT``: Perform 16-bit, optimized for CNN quantization.
-  ``INT8_TRANSFORMER_DEFAULT``: Perform 8-bit, optimized for transformer quantization.
-  ``INT16_TRANSFORMER_DEFAULT``: Perform 16-bit, optimized for transformer quantization.
-  ``INT8_CNN_ACCURATE``: Perform 8-bit, optimized for CNN quantization. Some advanced algorithms are applied to achieve higher accuracy but consume more time and memory space.
-  ``INT16_CNN_ACCURATE``: Perform 16-bit, optimized for CNN quantization. Some advanced algorithms are applied to achieve higher accuracy but consume more time and memory space.
-  ``INT8_TRANSFORMER_ACCURATE``: Perform 8-bit, optimized for transformer quantization. Some advanced algorithms are applied to achieve higher accuracy but consume more time and memory space.
-  ``INT16_TRANSFORMER_ACCURATE``: Perform 16-bit, optimized for transformer quantization. Some advanced algorithms are applied to achieve higher accuracy but consume more time and memory space.

AMD Quark for ONNX also provides more advanced default configurations to help you quantize models with more options.

-  ``UINT8_DYNAMIC_QUANT``: Perform dynamic activation, uint8 weight quantization.
-  ``XINT8``: Perform uint8 activation, int8 weight, optimized for NPU quantization.
-  ``XINT8_ADAROUND``: Perform uint8 activation, int8 weight, optimized for NPU quantization. The adaround fast finetune applies to preserve quantized accuracy.
-  ``XINT8_ADAQUANT``: Perform uint8 activation, int8 weight, optimized for NPU quantization. The adaquant fast finetune applies to preserve quantized accuracy.
-  ``S8S8_AAWS``: Perform int8 asymmetric activation, int8 symmetric weight quantization.
-  ``S8S8_AAWS_ADAROUND``: Perform int8 asymmetric activation, int8 symmetric weight quantization. The adaround fast finetune applies to preserve quantized accuracy.
-  ``S8S8_AAWS_ADAQUANT``: Perform int8 asymmetric activation, int8 symmetric weight quantization. The adaquant fast finetune applies to preserve quantized accuracy.
-  ``U8S8_AAWS``: Perform uint8 asymmetric activation int8 symmetric weight quantization.
-  ``U8S8_AAWS_ADAROUND``: Perform uint8 asymmetric activation, int8 symmetric weight quantization. The adaround fast finetune applies to preserve quantized accuracy.
-  ``U8S8_AAWS_ADAQUANT``: Perform uint8 asymmetric activation, int8 symmetric weight quantization. The adaquant fast finetune applies to preserve quantized accuracy.
-  ``S16S8_ASWS``: Perform int16 symmetric activation, int8 symmetric weight quantization.
-  ``S16S8_ASWS_ADAROUND``: Perform int16 symmetric activation, int8 symmetric weight quantization. The adaround fast finetune applies to preserve quantized accuracy.
-  ``S16S8_ASWS_ADAQUANT``: Perform int16 symmetric activation, int8 symmetric weight quantization. The adaquant fast finetune applies to preserve quantized accuracy.
-  ``A8W8``: Perform int8 symmetric activation, int8 symmetric weight quantization and optimize for deployment.
-  ``A16W8``: Perform int16 symmetric activation, int8 symmetric weight quantization and optimize for deployment.
-  ``U16S8_AAWS``: Perform uint16 asymmetric activation, int8 symmetric weight quantization.
-  ``U16S8_AAWS_ADAROUND``: Perform uint16 asymmetric activation, int8 symmetric weight quantization. The adaround fast finetune applies to preserve quantized accuracy.
-  ``U16S8_AAWS_ADAQUANT``: Perform uint16 asymmetric activation, int8 symmetric weight quantization. The adaquant fast finetune applies to preserve quantized accuracy.
-  ``BF16``: Perform BFloat16 activation, BFloat16 weight quantization.
-  ``BFP16``: Perform BFP16 activation, BFP16 weight quantization.
-  ``S16S16_MIXED_S8S8``: Perform int16 activation, int16 weight mix-precision quantization.

Customized Configurations
-------------------------

Besides the default configurations in AMD Quark for ONNX, you can also customize the quantization configuration like the following example:

.. toctree::
   :hidden:
   :caption: Advanced AMD Quark Features for PyTorch
   :maxdepth: 1

   Full List of Quantization Config Features <appendix_full_quant_config_features>

.. code-block:: python

   from quark.onnx import ModelQuantizer
   from quark.onnx.quantization import QConfig
   from quark.onnx.quantization.config.spec import QLayerConfig, Int8Spec
   from quark.onnx.quantization.config.data_type import Int16
   from quark.onnx.quantization.config.algorithm import CLEConfig, AdaRoundConfig

   input_model_path = "demo.onnx"
   quantized_model_path = "demo_quantized.onnx"
   calib_data_path = "calib_data"

   int8_config = QLayerConfig(activation=Int8Spec, weight=Int8Spec)
   cle_algo = CLEConfig(cle_steps=2)
   adaround_algo = AdaRoundConfig(learning_rate=0.1, num_iterations=1000)

   calib_data_reader = ImageDataReader(calib_data_path)
   quantization_config = QConfig(
       global_config=int8_config,
       specific_layer_config={Int16: ["/layer.0/Conv_0", "/layer.11/Conv_2"]},
       layer_type_config={Int16: ["MatMul"] None: ["Gemm"]},
       exclude=["/layer.2/Conv_1", "^/Conv/.*", (["start_node_1", "start_node_2"], ["end_node_1", "end_node_2"])],
       algo_config=[cle_algo, adaround_algo],
       use_external_data_format=False,
       **kwargs
   )
   quantizer = ModelQuantizer(quantization_config)
   quantizer.quantize_model(input_model_path, quantized_model_path, calib_data_reader)

.. toctree::
   :hidden:
   :maxdepth: 1

   Calibration methods <config/calibration_methods.rst>
   Calibration datasets <config/calibration_datasets.rst>
   Quantization Strategies <config/quantization_strategies.rst>
   Quantization Schemes <config/quantization_schemes.rst>
   Quantization Symmetry <config/quantization_symmetry.rst>
