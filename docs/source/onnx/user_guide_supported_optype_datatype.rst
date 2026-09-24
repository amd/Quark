.. Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.

Supported Data and Op Types
===========================

Supported Data Types
--------------------

Summary Table
~~~~~~~~~~~~~

+------------------------------------------------------------------------------+
| Supported Data Types                                                         |
+==============================================================================+
| Int8 / UInt8                                                                 |
+------------------------------------------------------------------------------+
| Int16 / UInt16                                                               |
+------------------------------------------------------------------------------+
| Int32 / UInt32                                                               |
+------------------------------------------------------------------------------+
| Float16                                                                      |
+------------------------------------------------------------------------------+
| BFloat16                                                                     |
+------------------------------------------------------------------------------+
| Float8(E4M3FN) / Float8(E5M2)                                                |
+------------------------------------------------------------------------------+
| BFP16                                                                        |
+------------------------------------------------------------------------------+
| MX4 / MX6 / MX9                                                              |
+------------------------------------------------------------------------------+
| MXFP8(E5M2) / MXFP8(E4M3) / MXFP6(E3M2) / MXFP6(E2M3) / MXFP4(E2M1) / MXINT8 |
+------------------------------------------------------------------------------+

You can see in the table there are many non integer data types that onnxruntime official operators do not support. In order to support these new features, we have developed several custom operators using onnxruntime's custom operation C APIs. Here are these ops and their specifications:

**ExtendedQuantizeLinear** - :doc:`specification <custom_operators/ExtendedQuantizeLinear>`

**ExtendedDequantizeLinear** - :doc:`specification <custom_operators/ExtendedDequantizeLinear>`

**BFPQuantizeDequantize** - :doc:`specification <custom_operators/BFPQuantizeDequantize>`

**MXQuantizeDequantize** - :doc:`specification <custom_operators/MXQuantizeDequantize>`

.. note::

   When installing on Windows, Visual Studio is required. The minimum version of Visual Studio is Visual Studio 2022. During the compilation process, there are two ways to use it:

1. **Use the Developer Command Prompt for Visual Studio**
   When installing Visual Studio, ensure that the Developer Command Prompt for Visual Studio is installed as well. Execute programs in the CMD window of the Developer Command Prompt for Visual Studio.

2. **Manually Add Paths to Environment Variables**
   Visual Studio's ``cl.exe``, ``MSBuild.exe``, and ``link.exe`` will be used. Ensure that the paths are added to the `PATH` environment variable. These programs are located in the Visual Studio installation directory. In the *Edit Environment Variables* window, click **New**, then paste the path to the folder containing ``cl.exe``, ``link.exe``, and ``MSBuild.exe``. Click **OK** on all windows to apply the changes.

1. Quantizing to Other Precision Levels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to the INT8/UINT8, the quark.onnx supports quantizing models to other data formats, including INT16/UINT16, INT32/UINT32, Float16 and BFloat16, which can provide better accuracy or be used for experimental purposes. The code below is an example for Int16. The table below shows all data type specs.

.. code:: python

    from quark.onnx import ModelQuantizer, QConfig, QLayerConfig, Int16Spec

    config = QConfig(global_config=QLayerConfig(input_tensors=Int16Spec(), weight=Int16Spec()))
    quantizer = ModelQuantizer(config)
    quantizer.quantize_model(model_input, model_output, calibration_data_reader)


+----------------------+------------------+
| Data Type            | Spec             |
+======================+==================+
| Int8                 | Int8Spec         |
+----------------------+------------------+
| UInt8                | UInt8Spec        |
+----------------------+------------------+
| Int16                | Int16Spec        |
+----------------------+------------------+
| UInt16               | UInt16Spec       |
+----------------------+------------------+
| Int32                | Int32Spec        |
+----------------------+------------------+
| UInt32               | UInt32Spec       |
+----------------------+------------------+
| BFloat16             | BFloat16Spec     |
+----------------------+------------------+
| BFP16                | BFP16Spec        |
+----------------------+------------------+
| MX4                  | MX4Spec          |
+----------------------+------------------+
| MX6                  | MX6Spec          |
+----------------------+------------------+
| MX9                  | MX9Spec          |
+----------------------+------------------+
| MXFP4E2M1            | MXFP4E2M1Spec    |
+----------------------+------------------+
| MXFP6E3M2            | MXFP6E3M2Spec    |
+----------------------+------------------+
| MXFP6E2M3            | MXFP6E2M3Spec    |
+----------------------+------------------+
| MXFP8E5M2            | MXFP8E5M2Spec    |
+----------------------+------------------+
| MXFP8E4M3            | MXFP8E4M3Spec    |
+----------------------+------------------+
| MXInt8               | MXInt8Spec       |
+----------------------+------------------+


.. note::

   BFP16 and MX data types use custom ops. When inference with ONNX Runtime, we need to register the custom op's so(Linux) or dll(Windows) file in the ORT session options.

.. code:: python

    import onnxruntime
    from quark.onnx import get_library_path

    device = 'CPU'
    providers = ['CPUExecutionProvider']

    # Also We can use the GPU configuration:
    # device='ROCM'
    # providers = ['ROCMExecutionProvider']
    # device='CUDA'
    # providers = ['CUDAExecutionProvider']

    sess_options = onnxruntime.SessionOptions()
    sess_options.register_custom_ops_library(get_library_path(device))
    session = onnxruntime.InferenceSession(onnx_model_path, sess_options, providers=providers)

2. Quantizing Float16 Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For models in Float16, we recommend setting ``ConvertFP16ToFP32`` to True in extra_options. This first converts your Float16 model to a Float32 model before quantization, reducing redundant nodes such as cast in the model.

.. code:: python

    from quark.onnx import ModelQuantizer, QConfig, QLayerConfig, Int8Spec

    config = QConfig(global_config=QLayerConfig(input_tensors=Int8Spec(), weight=Int8Spec()),
                     extra_options={"ConvertFP16ToFP32": True})
    quantizer = ModelQuantizer(config)
    quantizer.quantize_model(model_input, model_output, calibration_data_reader)


.. note::

   When using ``ConvertFP16ToFP32`` in quark.onnx, it requires onnxslim to simplify the ONNX model. Ensure that onnxslim is installed by using ``python -m pip install onnxslim``.

3. Quantizing to Standard FP8 (E4M3FN / E5M2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In addition to the Microscaling ``MXFP8`` data types (which use block-wise scales and the ``MXQuantizeDequantize`` custom op), quark.onnx also supports the *standard* per-tensor / per-channel FP8 formats defined in the ONNX standard: ``FLOAT8E4M3FN`` and ``FLOAT8E5M2``. These use the native ONNX ``QuantizeLinear`` / ``DequantizeLinear`` operators (no custom op required) with a symmetric MinMax scale ``scale = absmax / fp8_max`` and a zero-point of ``0``.

Standard FP8 is selected through ONNX Runtime's ``QuantType`` rather than a ``*Spec`` class. Use the legacy ``QuantizationConfig`` API:

.. code:: python

    from quark.onnx import ModelQuantizer
    from quark.onnx.quantization.config.config import Config
    from quark.onnx.quantization.config.legacy import QuantizationConfig
    from onnxruntime.quantization.calibrate import CalibrationMethod
    from onnxruntime.quantization.quant_utils import QuantFormat, QuantType

    # E4M3FN is exposed natively by ONNX Runtime's QuantType.
    quant_config = QuantizationConfig(
        calibrate_method=CalibrationMethod.MinMax,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QFLOAT8E4M3FN,
        weight_type=QuantType.QFLOAT8E4M3FN,
        op_types_to_quantize=["Conv", "Gemm", "MatMul"],
        per_channel=False,             # set True for per-channel weight scales
        extra_options={
            "ActivationSymmetric": True,   # FP8 zero-point is always 0
            "WeightsOnly": True,           # recommended for LLMs (weights-only)
        },
    )
    quantizer = ModelQuantizer(Config(global_quant_config=quant_config))
    quantizer.quantize_model(model_input, model_output, calibration_data_reader)

The two standard FP8 variants and how to request them:

+----------------+----------------------------+------------------------------------------------------------+
| Data Type      | Max finite value           | How to request                                             |
+================+============================+============================================================+
| Float8 E4M3FN  | 448.0                      | ``QuantType.QFLOAT8E4M3FN``                                |
+----------------+----------------------------+------------------------------------------------------------+
| Float8 E5M2    | 57344.0                    | quant type object with                                     |
|                |                            | ``tensor_type = onnx.TensorProto.FLOAT8E5M2``              |
+----------------+----------------------------+------------------------------------------------------------+

.. note::

   Standard FP8 ``QuantizeLinear`` / ``DequantizeLinear`` only load in ONNX Runtime at **opset >= 21**. quark.onnx automatically converts a lower-opset input model up to opset 21 before inserting FP8 Q/DQ nodes, so no manual opset bump is required. Only the ``CalibrationMethod.MinMax`` and ``CalibrationMethod.Distribution`` methods are supported for FP8.

.. note::

   For LLMs, weights-only FP8 (``WeightsOnly=True``) is strongly recommended. Quantizing activations to FP8 severely degrades attention accuracy. In weights-only mode only weight tensors receive Q/DQ nodes while activations stay in Float16/Float32.

Supported Op Type
-----------------

.. _quark-onnx-supported-ops:

Summary Table
~~~~~~~~~~~~~

**Note:** For built-in configs except those with block floating-point data types, the extra option ``ForceQuantizeNoInputCheck`` is set to ``True`` by default. In that case, ops listed below as "quantized only when input is quantized" are *always* quantized (their inputs are quantized and quantized outputs are produced). For custom configs with ``ForceQuantizeNoInputCheck=False``, those ops follow the input-dependent behavior.

Table: List of Quark ONNX Supported Quantized Ops

+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Supported Ops         | Comments                                                                                                                                                                                                  |
+=======================+===========================================================================================================================================================================================================+
| Add                   |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ArgMax                |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| AveragePool           | Quantized only when its input is quantized (or always when ForceQuantizeNoInputCheck=True).                                                                                                               |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| BatchNormalization    | By default, the "optimize_model" parameter will fuse BatchNormalization to Conv/ConvTranspose/Gemm. For standalone BatchNormalization, quantization is supported only for NPU_CNN platforms by converting |
|                       | BatchNormalization to Conv.                                                                                                                                                                               |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Clip                  | Quantized only when its input is quantized (or always when ForceQuantizeNoInputCheck=True).                                                                                                               |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Concat                |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Conv                  |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ConvTranspose         |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| DepthToSpace          | Quantization is supported only for NPU_CNN platforms.                                                                                                                                                     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Div                   | Quantization is supported only for NPU_CNN platforms.                                                                                                                                                     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Erf                   | Quantization is supported only for NPU_CNN platforms.                                                                                                                                                     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Gather                |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Gemm                  |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| GlobalAveragePool     |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| HardSigmoid           | Quantization is supported only for NPU_CNN platforms.                                                                                                                                                     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| InstanceNormalization |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| LayerNormalization    | Supported for opset>=17. Quantized only when its input is quantized (or always when ForceQuantizeNoInputCheck=True).                                                                                      |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| LeakyRelu             |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| LpNormalization       | Quantization is supported only for NPU_CNN platforms.                                                                                                                                                     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| MatMul                |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Min                   | Quantization is supported only for NPU_CNN platforms.                                                                                                                                                     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Max                   | Quantization is supported only for NPU_CNN platforms.                                                                                                                                                     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| MaxPool               | Quantized only when its input is quantized (or always when ForceQuantizeNoInputCheck=True).                                                                                                               |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Mul                   |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Pad                   |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| PRelu                 | Quantization is supported only for NPU_CNN platforms.                                                                                                                                                     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ReduceMean            | Quantization is supported only for NPU_CNN platforms.                                                                                                                                                     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Relu                  | Quantized only when its input is quantized (or always when ForceQuantizeNoInputCheck=True).                                                                                                               |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Reshape               | Quantized only when its input is quantized (or always when ForceQuantizeNoInputCheck=True).                                                                                                               |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Resize                |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Slice                 | Quantization is supported only for NPU_CNN platforms.                                                                                                                                                     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Sigmoid               |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Softmax               |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| SpaceToDepth          | Quantization is supported only for NPU_CNN platforms.                                                                                                                                                     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Split                 |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Squeeze               | Quantized only when its input is quantized (or always when ForceQuantizeNoInputCheck=True).                                                                                                               |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Sub                   | Quantization is supported only for NPU_CNN platforms.                                                                                                                                                     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Tanh                  | Quantization is supported only for NPU_CNN platforms.                                                                                                                                                     |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Transpose             | Quantized only when its input is quantized (or always when ForceQuantizeNoInputCheck=True).                                                                                                               |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Unsqueeze             | Quantized only when its input is quantized (or always when ForceQuantizeNoInputCheck=True).                                                                                                               |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Where                 |                                                                                                                                                                                                           |
+-----------------------+-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

.. toctree::
   :hidden:
   :caption: Custom Operators
   :maxdepth: 1

   ExtendedQuantizeLinear <custom_operators/ExtendedQuantizeLinear>
   ExtendedDequantizeLinear <custom_operators/ExtendedDequantizeLinear>
   ExtendedInstanceNormalization <custom_operators/ExtendedInstanceNormalization>
   ExtendedLSTM <custom_operators/ExtendedLSTM>
   BFPQuantizeDequantize <custom_operators/BFPQuantizeDequantize>
   MXQuantizeDequantize <custom_operators/MXQuantizeDequantize>
