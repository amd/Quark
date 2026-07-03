.. Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

Quantization Configuration Serialization and Deserialization
============================================================

Quantization Configuration Serialization
----------------------------------------

All AMD Quark ONNX's quantization configs like `QConfig`, `QLayerConfig`, `QTensorConfig` and `AlgoConfig` can be serialized to dictionaries. The following code is an example of how to serialize quantization configs.

.. code-block:: python

   from quark.onnx import CLEConfig, Int16Spec, ModelQuantizer, QConfig, QLayerConfig, XInt8Spec

   cle_algo = CLEConfig(cle_steps=2)
   quant_config = QConfig(
       global_config=QLayerConfig(input_tensors=XInt8Spec(), weight=XInt8Spec()),
       specific_layer_config={
           QLayerConfig(input_tensors=Int16Spec(), weight=Int16Spec()): ["/conv1/Conv", "/conv2/Conv"]
       },
       layer_type_config={None: ["Gemm"]},
       algo_config=[cle_algo],
       extra_options={"SimplifyModel": False, "Int32Bias": False},
   )

   quant_config_dict = quant_config.to_dict()

The serialized dictionary of the above quantization configuration is shown below. The serialization of `QLayerConfig`, `QTensorConfig` and `AlgoConfig` follows the same approach.

.. code-block:: python

   quant_config_dict = {
       "global_config": {
           "input_tensors": {
               "symmetric": True,
               "scale_type": "ScaleType.PowerOf2",
               "calibration_method": "CalibMethod.MinMSE",
               "quant_granularity": "QuantGranularity.Tensor",
               "data_type": "Int8",
           },
           "weight": {
               "symmetric": True,
               "scale_type": "ScaleType.PowerOf2",
               "calibration_method": "CalibMethod.MinMSE",
               "quant_granularity": "QuantGranularity.Tensor",
               "data_type": "Int8",
           },
       },
       "specific_layer_config": [
           [
               {
                   "input_tensors": {
                       "symmetric": True,
                       "scale_type": "ScaleType.Float32",
                       "calibration_method": "CalibMethod.Percentile",
                       "quant_granularity": "QuantGranularity.Tensor",
                       "data_type": "Int16",
                   },
                   "weight": {
                       "symmetric": True,
                       "scale_type": "ScaleType.Float32",
                       "calibration_method": "CalibMethod.Percentile",
                       "quant_granularity": "QuantGranularity.Tensor",
                       "data_type": "Int16",
                   },
               },
               ["/conv1/Conv", "/conv2/Conv"],
           ]
       ],
       "layer_type_config": [[None, ["Gemm"]]],
       "exclude": [],
       "algo_config": [{"name": "cle", "cle_steps": 2}],
       "use_external_data_format": False,
       "extra_options": {"SimplifyModel": False, "Int32Bias": False},
   }

Quantization Configuration Deserialization
------------------------------------------

Deserialization is the inverse process of serialization. All AMD Quark ONNX's quantization configs like `QConfig`, `QLayerConfig`, `QTensorConfig` and `AlgoConfig` can be deserialized from dictionaries. The following code is an example of how to deserialize quantization configs.

.. code-block:: python

   from quark.onnx import QConfig

   # You can take the quant_config_dict shown above as example
   quant_config = QConfig.from_dict(quant_config_dict)

Next, this quantization config can be used directly for quantization. The following code is an example of how to quantize models. For more details about how to quantize an ONNX model, see :doc:`Getting started: Quark for ONNX <../basic_usage_onnx>`.

.. code-block:: python

    from quark.onnx import ModelQuantizer

    input_model_path = "models/resnet50-v1-12.onnx"
    quantized_model_path = "models/resnet50-v1-12_quantized.onnx"
    calib_data_path = "calib_data"
    model_input_name = get_model_input_name(input_model_path)
    calib_data_reader = ImageDataReader(calib_data_path, model_input_name)

    quantizer = ModelQuantizer(quant_config)
    quantizer.quantize_model(input_model_path, quantized_model_path, calib_data_reader)
