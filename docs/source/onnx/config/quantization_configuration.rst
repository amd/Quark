Calibration Methods
===================

AMD Quark for ONNX supports these types of calibration methods:

MinMax Calibration Method
~~~~~~~~~~~~~~~~~~~~~~~~~
The MinMax calibration method computes the quantization parameters based on the running minimum and maximum values. This method uses the tensor min/max statistics to compute the quantization parameters. The module records the running minimum and maximum of incoming tensors and uses these statistics to compute the quantization parameters.

Percentile Calibration Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The Percentile calibration method, often used in robust scaling, involves scaling features based on percentile information from a static histogram, rather than using the absolute minimum and maximum values. This method is particularly useful for managing outliers in data.

Layer-wise Percentile Calibration Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Layerwise percentile calibration offers a more granular approach than standard percentile calibration. Instead of using global min/max values or a single percentile for the whole model, it collects weight and/or input_tensors statistics for each layer and selects clipping thresholds from those per-layer distributions. This produces more appropriate quantization ranges that better reflect each layer's unique dynamic range.

MSE Calibration Method
~~~~~~~~~~~~~~~~~~~~~~
The MSE (Mean Squared Error) calibration method involves performing calibration by minimizing the mean squared error between the predicted outputs and the actual outputs. This method is typically used in regression contexts where the goal is to adjust model parameters or data transformations to reduce the average squared difference between estimated values and the true values. MSE calibration helps in refining model accuracy by fine-tuning predictions to be as close as possible to the real data points.

Entropy Calibration Method
~~~~~~~~~~~~~~~~~~~~~~~~~~
Entropy calibration uses an information-theoretic approach, aiming to preserve the core of the data distribution rather than relying on simple extreme based methods that are overly sensitive to outliers. It works by computing the KL divergence or relative entropy between the original distribution and the quantized distribution, then selects the clipping threshold that minimizes this divergence.

Distribution Calibration Method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The Distribution calibration computes quantization parameters by examining the overall distribution of tensor values. It constructs a histogram to capture how values are spread and then selects quantization ranges that best reflect the distribution's shape, helping to preserve data fidelity—especially when the data has non-uniform patterns or long tails. Note: only FP8 data type is supported under this calibration method.

Data Types
==========

Quark supports the ONNX data types listed below. For each Quark data type, there is a corresponding ONNX Tensor Proto data type and a mapped ONNX Runtime quantization format.

+--------------+----------------+-----------------+
|   Data Type  |     Min Val    |      Max Val    |
+==============+================+=================+
|     Int4     |       -8       |        7        |
+--------------+----------------+-----------------+
|     UInt4    |        0       |        15       |
+--------------+----------------+-----------------+
|     Int8     |      -128      |       127       |
+--------------+----------------+-----------------+
|     UInt8    |        0       |       255       |
+--------------+----------------+-----------------+
|     Int16    |     -32,768    |      32,767     |
+--------------+----------------+-----------------+
|    UInt16    |         0      |      65,535     |
+--------------+----------------+-----------------+
|     Int32    |       -2³¹     |     2³¹ - 1     |
+--------------+----------------+-----------------+
|     UInt32   |        0       |     2³² - 1     |
+--------------+----------------+-----------------+
|    Float16   |     -65,504    |      65,504     |
+--------------+----------------+-----------------+
|   BFloat16   |      2⁻¹²⁶     |(2 - 2⁻⁷) x 2¹²⁷ |
+--------------+----------------+-----------------+
|     BFP16    |        /       |         /       |
+--------------+----------------+-----------------+
|      MX4     |        /       |         /       |
+--------------+----------------+-----------------+
|      MX6     |        /       |         /       |
+--------------+----------------+-----------------+
|      MX9     |        /       |         /       |
+--------------+----------------+-----------------+
|    MX Int8   |        /       |         /       |
+--------------+----------------+-----------------+
|  MXFP4 E2M1  |      -6.0      |       6.0       |
+--------------+----------------+-----------------+
|  MXFP6 E3M2  |     -28.0      |      28.0       |
+--------------+----------------+-----------------+
|  MXFP6 E2M3  |      -7.5      |       7.5       |
+--------------+----------------+-----------------+
|  MXFP8 E5M2  |    -57344.0    |     57344.0     |
+--------------+----------------+-----------------+
|  MXFP8 E4M3  |     -448.0     |      448.0      |
+--------------+----------------+-----------------+

Scale Type
==========

ScaleType specifies how the quantization scale is represented and determines the numeric format and arithmetic used for quantize/dequantize and trading off accuracy for hardware efficiency. Quark supports three types of scale: Float32, PowerOf2, and Int16.

Quant Granularity
=================
Quantization granularity specifies the scope over which scale/zero-point parameters are shared when mapping floating-point values to integers. Choosing granularity controls accuracy, memory overhead, and runtime complexity. Quark provide three levels of granularity: tensor, channel, and group.

Quantization Symmetry
=====================

Symmetric and asymmetric quantization are two ways to convert floating-point values 𝑥 into low-precision integer codes 𝑞. Both use a positive scaling factor scale, but they differ in whether including an integer zero-point z or not. Symmetric quantization is simpler and works well for zero-centered distributions; asymmetric is more flexible when the data range doesn't straddle zero.

``Symmetric``: the representation is centered at zero (zero-point 𝑧 = 0), we get 𝑞 = round( 𝑥 / scale)

``Asymmetric``: a nonzero integer zero-point shifts the mapped range to better fit non-zero-centered data, we get 𝑞 = round( 𝑥 / scale ) + 𝑧


Define A Configuration
======================

.. code-block:: python

    from quark.onnx import CalibMethod, QConfig, QLayerConfig, ModelQuantizer, Int8

    """
    This is the default setting of Int8 configuration and demonstrates how symmetry strategy, scale type, granularity, and data type are applied.
    """
    class Int8Spec(QTensorConfig):
        def __init__(
            self,
            symmetric: bool = True,
            scale_type: ScaleType = ScaleType.Float32,
            calibration_method: CalibMethod = CalibMethod.Percentile,
            quant_granularity: QuantGranularity = QuantGranularity.Tensor,
            data_type: type[BaseDataType] = Int8,
        ):
            super().__init__(symmetric, scale_type, calibration_method, quant_granularity, data_type)

    """
    Customize the Int8 calibration method to MinMax. Other supported options include: CalibMethod.MinMSE, CalibMethod.Percentile, CalibMethod.LayerwisePercentile, CalibMethod.Entropy, CalibMethod.Distribution
    """

    input_tensors_spec = Int8Spec(calibration_method=CalibMethod.MinMax)
    weight_spec = Int8Spec(calibration_method=CalibMethod.MinMax)

    config = QConfig(
        global_config=QLayerConfig(input_tensors=input_tensors_spec, weight=weight_spec)
    )

    quantizer = ModelQuantizer(config)
