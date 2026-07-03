.. Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

Introduction
============

.. note::

    In this documentation, **AMD Quark** is sometimes referred to simply as **"Quark"** for ease of reference. When you  encounter the term "Quark" without the "AMD" prefix, it specifically refers to the AMD Quark quantizer unless otherwise stated. Please do not confuse it with other products or technologies that share the name "Quark."

BFloat16 (Brain Floating Point 16) is a floating-point data format used in deep learning to reduce memory usage and computation while maintaining sufficient numerical precision. Unlike other quantization formats like INT8 or FP16, BF16 maintains the same range as FP32 but reduces precision, making it particularly useful for training and inference in neural networks.

AMD accelerators like latest CPU, NPU and GPU devices support BF16 natively, enabling faster matrix operations and reducing latency. In this tutorial, we will explain how to quantize a model into BF16 using AMD Quark.

BF16 quantization in AMD Quark for ONNX
---------------------------------------

Below are examples of how to enable BF16 quantization.

Convert to Cast Format
~~~~~~~~~~~~~~~~~~~~~~

The bfloat16 conversion is implemented by inserting Cast operations to convert from float32/float16 to bfloat16. A pair of Cast nodes will be inserted between every two nodes. The first Cast converts float32/float16 to bfloat16, and the second Cast converts bfloat16 back to float32/float16.

FP32->BF16
^^^^^^^^^^

.. code-block:: bash

    python -m quark.onnx.tools.convert_fp32_to_bf16 --input $FLOAT_32_ONNX_MODEL_PATH --output $BFLOAT_16_ONNX_MODEL_PATH --format with_cast

FP16->BF16
^^^^^^^^^^

.. code-block:: bash

    python -m quark.onnx.tools.convert_fp16_to_bf16 --input $FLOAT_16_ONNX_MODEL_PATH --output $BFLOAT_16_ONNX_MODEL_PATH --format with_cast

Convert Directly
~~~~~~~~~~~~~~~~

The float/float16 model is directly converted to bfloat16 and only the input and output are remained as float/float16. It only supports by onnxruntime-gpu.

FP32->BF16
^^^^^^^^^^

.. code-block:: bash

    python -m quark.onnx.tools.convert_fp32_to_bf16 --input $FLOAT_32_ONNX_MODEL_PATH --output $BFLOAT_16_ONNX_MODEL_PATH --format bf16

FP16->BF16
^^^^^^^^^^

.. code-block:: bash

    python -m quark.onnx.tools.convert_fp16_to_bf16 --input $FLOAT_16_ONNX_MODEL_PATH --output $BFLOAT_16_ONNX_MODEL_PATH --format bf16

Convert to Float32 Format
~~~~~~~~~~~~~~~~~~~~~~~~~

The bfloat16 conversion is implemented by that all bfloat16 weights are stored as float32 format.

FP32->BF16
^^^^^^^^^^

.. code-block:: bash

    python -m quark.onnx.tools.convert_fp32_to_bf16 --input $FLOAT_32_ONNX_MODEL_PATH --output $BFLOAT_16_ONNX_MODEL_PATH --format simulate_bf16

FP16->BF16
^^^^^^^^^^

.. code-block:: bash

    python -m quark.onnx.tools.convert_fp16_to_bf16 --input $FLOAT_16_ONNX_MODEL_PATH --output $BFLOAT_16_ONNX_MODEL_PATH --format simulate_bf16

Convert to Customized QDQ
~~~~~~~~~~~~~~~~~~~~~~~~~

The bfloat16 conversion is implemented by inserting customized QDQ of bfloat16.

FP32->BF16
^^^^^^^^^^

.. code-block:: bash

    python -m quark.onnx.tools.convert_fp32_to_bf16 --input $FLOAT_32_ONNX_MODEL_PATH --output $BFLOAT_16_ONNX_MODEL_PATH --format customqdq

FP16->BF16
^^^^^^^^^^

.. code-block:: bash

    python -m quark.onnx.tools.convert_fp16_to_bf16 --input $FLOAT_16_ONNX_MODEL_PATH --output $BFLOAT_16_ONNX_MODEL_PATH --format customqdq

.. note::
    When inference with ONNXRuntime, you need to register the custom OPs so(Linux) or dll(Windows) file in the ORT session options.

.. code:: python

    import onnxruntime
    from quark.onnx import get_library_path

    if 'ROCMExecutionProvider' in onnxruntime.get_available_providers():
        device = 'ROCM'
        providers = ['ROCMExecutionProvider']
    elif 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
        device = 'CUDA'
        providers = ['CUDAExecutionProvider']
    else:
        device = 'CPU'
        providers = ['CPUExecutionProvider']

    sess_options = onnxruntime.SessionOptions()
    sess_options.register_custom_ops_library(get_library_path(device))
    session = onnxruntime.InferenceSession(onnx_model_path, sess_options, providers=providers)

How to Further Improve the Accuracy for BF16 Quantization?
----------------------------------------------------------

You can finetune the quantized model to further improve the accuracy of BF16 quantization.
The Fast Finetuning function in AMD Quark for ONNX includes two algorithms: AdaRound and AdaQuant.
There is no explicit rounding in BF16 quantization, so only AdaQuant can be used.

.. code:: python

    from quark.onnx import AdaQuantConfig

    input_tensors_spec = BFloat16Spec()
    weight_spec = BFloat16Spec()
    algo_conf = [AdaQuantConfig(num_iterations=1000, learning_rate=1e-6)]
    config = QConfig(global_config=QLayerConfig(input_tensors=input_tensors_spec, weight=weight_spec),
                    algo_config=algo_conf)
