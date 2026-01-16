.. Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.

Troubleshooting
===============

AMD Quark for ONNX
------------------

Storage Errors
~~~~~~~~~~~~~~

**Issue 1**:

Error of "No Space on Device"

**Solution**:

This error is caused by the /tmp directory is out of space. Set ``TmpDir`` as your directory with sufficient available space like ``/home/xxx/tmp/`` in extra_options. For more detailed information, see :doc:`Full List of Quantization Config Features <appendix_full_quant_config_features>`.

**Issue 2**:

The process was automatically terminated during calibration due to an out-of-memory condition.

**Solution**:

This issue is caused by insufficient memory. If you are using amd-quark version earlier than 0.11, please upgrade to version 0.11 or later. Starting from version 0.11, a parameter named ``CalibOptimizeMem`` was introduced. When the calibration method is MinMSE or LayerwisePercentile, setting this parameter to True can effectively alleviate memory-related issues. Alternatively, you may resolve this issue by using a machine with more available memory.

Model inference Errors
~~~~~~~~~~~~~~~~~~~~~~

**Issue 1**:

Error of "ValueError:Message onnx.ModelProto exceeds maximum protobuf size of 2GB"

**Solution**:

This error is caused by the input model size exceeding 2GB. Set ``use_external_data_format`` as True in QConfig.

**Issue 2**:

Error of "index: 1 Got: 224 Expected: 3"

**Solution**:

This is usually caused by a mismatch between the input data shape and the model’s expected shape. Please check the input data shape format. If the error is caused by the calibration data is NHWC and the shape of model input is NCHW. Set ``ConvertNCHWToNHWC`` as True in extra_options. For more detailed information, see :doc:`Tools <tools>` or :doc:`Full List of Quantization Config Features <appendix_full_quant_config_features>`.

**Issue 3**:

Error of "onnxruntime.capi.onnxruntime_pybind11_state.InvalidGraph: [ONNXRuntimeError] : 10 : INVALID_GRAPH : Load model from demo.onnx failed:This is an invalid model. In Node, ("", BFPQuantizeDequantize, "com.amd.quark", -1) : ("input": tensor(float),) -> ("out": tensor(float),)"

**Solution**:

This is usually caused by that you do not register customized ops when you infer a BF16/BFP16/MX model. The solution is like below:

.. code-block:: python

    import onnxruntime
    from quark.onnx.operators.custom_ops import get_library_path

    so = onnxruntime.SessionOptions()
    so.register_custom_ops_library(get_library_path())
    ort_session = onnxruntime.InferenceSession(onnx_model_path, so)

Quantization Error for Detection Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue 1**:

Error of "onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException: [ONNXRuntimeError] : 6 : RUNTIME_EXCEPTION : Non-zero status code returned while running Reshape node."

**Solution**:

One possible cause is that for networks with an ROI head, such as Mask R-CNN or Faster R-CNN, quantization errors might arise if ROIs are not generated in the network.
Use quark.onnx.PowerOfTwoMethod.MinMSE or quark.onnx.CalibrationMethod.Percentile quantization and perform inference with real data.


Quantization Accuracy
~~~~~~~~~~~~~~~~~~~~~

**Issue 1**:

What are the most commonly used methods to improve quantization accuracy?

**Solution**:

The most common methods to improve quantization accuracy are AdaRound and AdaQuant. And GPU are recommended for Faster AdaRound and AdaQuant. For more detailed information, see :doc:`AdaRound and AdaQuant <accuracy_algorithms/ada>`.

**Issue 2**:

How can the quantization accuracy of YOLO models be improved?

**Solution**:

The key to improving YOLO model accuracy is to exclude the post-processing subgraph. You can use ``exclude`` to exclude subgraphs. For more detailed information, see :doc:`Full List of Quantization Config Features <appendix_full_quant_config_features>`.

**Issue 3**:

How to improve a quantized model's accuracy from other perspectives?

**Solution**:

a. Experiment with the ``LayerwisePercentile`` calibration method if ``Percentile`` does not work well.

b. Try different calibration datasets. In particular, include more samples whose data distribution closely matches that of the test dataset.

c. Try to quantize only one operation type like MatMul or Conv to find whether there are some operation types that cause the quantization accuracy drop. For example, Set ``OpTypesToQuantize`` as ["MatMul", "Conv", "Gemm"] in extra_options. For more detailed information, see :doc:`Full List of Quantization Config Features <appendix_full_quant_config_features>`.

**Issue 4**:

How can we determine the highest achievable accuracy of a quantized model under a specific setting (e.g., a A8W8 config)?

**Solution**:

Use higher quantization precision (e.g., a A16W8 or A8W16 config) as a reference to establish the upper bound of achievable accuracy under the given configuration.


Quantization Acceleration
~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue 1**:

The calibration process is very slow, how can we accelerate it and what should we pay attention to?

**Solution**:

During calibration, multiple calibration datasets must be iterated over, and for each dataset the tensor ranges need to be collected and computed. To speed up this process, you can set ``CalibWorkerNum`` in ``extra_options`` to enable parallel computation and improve overall performance.

**Notes**:

A larger ``CalibWorkerNum`` consumes more memory. Avoid setting it too high to prevent potential OOM (Out of Memory) issues.

Parallel computation introduces additional overhead for managing the pipeline, so a larger value is not always better.

If sufficient memory is available, we recommend setting ``CalibWorkerNum`` to 8 or 16.

**Issue 2**:

The fine-tuning process is very slow, how can we accelerate it and what should we pay attention to?

**Solution**:

a. Use ``mem_opt_level`` = 1 for faster performance (default).

b. With ``mem_opt_level`` = 2, set ``num_workers`` > 1 (e.g., 8) and enable ``pin_memory`` = True to speed up data transfer.

c. Reuse calibration results with ``TensorsRangeFile`` when fine-tuning the same model using different hyperparameters.

d. Use more powerful GPUs (e.g., AMD MI350 performs better than MI300 or MI250).

**Notes**:

Setting ``mem_opt_level`` = 1 is faster but consumes more memory; ``mem_opt_level`` = 2 saves memory but is slower.

Increasing ``num_workers`` improves throughput but may increase memory usage and CPU contention. Avoid excessively large values to prevent OOM or runtime failures.


Quantization Config and Customized Deployment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Issue 1**:

Does XINT8 refer to INT8? What's the difference between XINT8 and A8W8?

**Solution**:

XINT8 and A8W8 are both INT8 Quantization. XINT8 and A8W8 are both very common quantization configurations in our Quark ONNX quantizer. A8W8 uses symmetric INT8 activation and weights quantization with float scales. XINT8 uses symmetric INT8 activation and weights quantization with power-of-two scales. XINT8 usually has greater advantages in hardware acceleration. For more detailed information about XINT8, see :doc:`Power-of-Two Scales (XINT8) Quantization <../supported_accelerators/ryzenai/tutorial_xint8_quantize>`. For more detailed information about A8W8, see :doc:`Float Scales (A8W8 and A16W8) Quantization <../supported_accelerators/ryzenai/tutorial_a8w8_and_a16w8_quantize>`.

**Issue 2**:

How to convert all BatchNormalization operations to Conv operations?

**Solution**:

Set ``ConvertBNToConv`` as True. For more detailed information, see :doc:`Full List of Quantization Config Features <appendix_full_quant_config_features>`.

**Issue 3**:

How to convert all Sigmoid operations to HardSigmoid operations?

**Solution**:

Set ``ConvertSigmoidToHardSigmoid`` as True. For more detailed information, see :doc:`Full List of Quantization Config Features <appendix_full_quant_config_features>`.
