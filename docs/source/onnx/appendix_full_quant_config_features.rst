.. Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.

Full List of Quantization Configuration Features
================================================

Overview
--------

It's very simple to quantize a model using the ONNX quantizer of Quark, only a few straightforward Python statements:

.. code:: python

    from quark.onnx import ModelQuantizer, QConfig, QLayerConfig, Int8Spec

    config = QConfig(global_config=QLayerConfig(input_tensors=Int8Spec(), weight=Int8Spec()))
    quantizer = ModelQuantizer(config)
    quantizer.quantize_model(model_input, model_output, calibration_data_reader)


As shown in the code, just create a quantization configuration and use it to initialize a quantizer, and then call the quantizer's *quantize_model()* API, which has 3 main parameters:

*  **model_input**: (String or ModelProto) This parameter specifies the file path of the model that is to be quantized. When a file path cannot be specified, the loaded ModelProto can also be passed in directly.
*  **model_output**: (Optional String) This parameter specifies the file path where the quantized model will be saved. You can leave it unspecified (it will default to None), and the ModelProto format quantized model will be returned by the API.
*  **calibration_data_reader**: (Optional Object) This parameter is a calibration data reader that enumerates the calibration data and generates inputs for the original model. You can leave it unspecified (it will default to None), and simply enable *UseRandomData* in extra options of quantization configuration to use random data for calibration.

The next section will provide a detailed list of all parameters in the quantization configuration.



Extra Options — Categorized Summary
------------------------------------

The table below groups all ``extra_options`` keys into functional categories.

.. list-table::
   :header-rows: 1
   :widths: 3 20 30 47

   * - #
     - Category
     - Description
     - Parameters
   * - 1
     - Preprocessing & Graph Optimization
     - Model simplification, operator fusion (LayerNorm, Gelu, InstanceNorm), BatchNorm folding, and format/layout conversion steps applied to the graph before quantization begins.
     - ``PreprocessYAML``, ``SkipPreprocess``, ``OptimizeModel``, ``SimplifyModel``, ``SimplifyModelOptions``, ``ConvertFP16ToFP32``, ``ConvertNCHWToNHWC``, ``ConvertOpsetVersion``, ``ConvertBNToConv``, ``ConvertReduceMeanToGlobalAvgPool``, ``SplitLargeKernelPool``, ``ConvertSplitToSlice``, ``ConvertClipToRelu``, ``FuseInstanceNorm``, ``FuseL2Norm``, ``FuseGelu``, ``FuseLayerNorm``, ``FoldBatchNorm``, ``FoldRelu``, ``ReplaceClip6Relu``
   * - 2
     - Quantization Target Selection
     - Determines the scope of quantization: which operator types, named nodes, or subgraphs are targeted, and whether to quantize weights only or all tensors.
     - ``OpTypesToQuantize``, ``NodesToQuantize``, ``ExtraOpTypesToQuantize``, ``QuantizeAllOpTypes``, ``WeightsOnly``, ``MatMulConstBOnly``, ``EnableSubgraph``
   * - 3
     - QDQ Node Management
     - Controls insertion style of QuantizeLinear/DequantizeLinear (QDQ) pairs and removal of redundant QDQ nodes between adjacent operators, especially for DPU/NPU fusion patterns.
     - ``AddQDQPairToWeight``, ``ForceQuantizeNoInputCheck``, ``DedicatedQDQPair``, ``EnableDualQuantNodePairs``, ``OpTypesToExcludeOutputQuantization``, ``NodesToExcludeOutputQuantization``, ``QDQOpTypePerChannelSupportToAxis``, ``RemoveQDQConvClip``, ``RemoveQDQConvRelu``, ``RemoveQDQConvLeakyRelu``, ``RemoveQDQConvPRelu``, ``RemoveQDQConvGelu``, ``RemoveQDQMulAdd``, ``RemoveQDQBetweenOps``, ``RemoveQDQInstanceNorm``, ``RemoveFusedQDQ``
   * - 4
     - Calibration Configuration
     - Tunes calibration behaviour: range symmetry, moving-average smoothing, percentile thresholds, random-data input, per-tensor overrides, parallelism, memory/disk trade-offs, and save-restore checkpointing.
     - ``CalibTensorRangeSymmetric``, ``CalibMovingAverage``, ``CalibMovingAverageConstant``, ``Percentile``, ``LWPMetric``, ``PercentileCandidates``, ``UseRandomData``, ``RandomDataReaderInputShape``, ``RandomDataReaderInputDataRange``, ``MinMSEModePof2Scale``, ``WeightCalibrateMethod``, ``MinMSEModeFloatScale``, ``CalibDataSize``, ``CalibWorkerNum``, ``CalibOptimizeMem``, ``CalibOptimizeDisk``, ``SaveTensorHistFig``, ``CalibPassthroughOpTypes``, ``TensorQuantOverrides``, ``SaveAndRestore``
   * - 5
     - Bias Handling
     - Specifies whether and how bias tensors are quantized (int32, int16, or same as weight) and whether the bias scale is aligned to input × weight scale.
     - ``QuantizeBias``, ``Int32Bias``, ``Int16Bias``, ``AdjustBiasScale``
   * - 6
     - Scale & Numeric Type
     - Controls scale representation format (float32, float16, int16/power-of-two) and whether activations or weights are mapped to a reduced numeric range before casting.
     - ``Int16Scale``, ``QuantizeFP16``, ``UseFP32Scale``, ``ActivationScaled``, ``WeightScaled``, ``UseUnsignedReLU``
   * - 7
     - DPU/NPU Hardware Adaptation
     - Replaces operators with DPU-compatible approximations (SimulateDPU) and iteratively adjusts quantization positions (shift, alignment) to satisfy NPU hardware constraints.
     - ``SimulateDPU``, ``ConvertLeakyReluToDPUVersion``, ``ConvertSigmoidToHardSigmoid``, ``ConvertHardSigmoidToDPUVersion``, ``ConvertAvgPoolToDPUVersion``, ``ConvertClipToDPUVersion``, ``ConvertReduceMeanToDPUVersion``, ``ConvertSoftmaxToDPUVersion``, ``NPULimitationCheck``, ``MaxLoopNum``, ``AdjustShiftCut``, ``AdjustShiftBias``, ``AdjustShiftRead``, ``AdjustShiftWrite``, ``AdjustHardSigmoid``, ``AdjustShiftSwish``, ``AlignConcat``, ``AlignPool``, ``AlignPad``, ``AlignSlice``, ``AlignTranspose``, ``AlignReshape``
   * - 8
     - BF16 / BFP / MX Quantization
     - Options specific to floating-point block quantization formats (BFloat16, BFP16, MX): boundary clipping, replacing QDQ with Cast for inference speed, block-axis refinement, and Vaiml compiler export.
     - ``BF16WithClip``, ``BF16QDQToCast``, ``RefineBlockAxis``, ``EnableVaimlBF16``
   * - 9
     - MatMul NBits (LLM Weight-Only)
     - Enables n-bit (typically 4-bit) weight-only quantization for MatMul ops, with algorithm choice (DEFAULT / GPTQ / HQQ), group size, and accuracy-level sub-parameters.
     - ``UseMatMulNBits``, ``MatMulNBitsParams``
   * - 10
     - Runtime & Infrastructure
     - Configures ORT execution providers for calibration, initializer deduplication, custom op libraries, fixed output shapes, temp directories, and cryptographic protection of model data.
     - ``ExecutionProviders``, ``RemoveInputInit``, ``CopySharedInit``, ``CopyBiasInit``, ``AlignEltwiseQuantType``, ``FixShapes``, ``TmpDir``, ``UserCustomOpLibPath``, ``EncryptionAlgorithm``, ``CryptoMode``, ``FillAllValueInfo``
   * - 11
     - Debug, Logging & Evaluation
     - Controls log verbosity and debug output, and optionally evaluates quantization quality by computing cosine similarity and L2 loss against the original float model.
     - ``DebugMode``, ``PrintSummary``, ``IgnoreWarnings``, ``EvalMetrics``, ``EvalDataReader``

Quantization Configuration
--------------------------

.. code:: python

    from quark.onnx import QLayerConfig, Int8Spec, QConfig

    quant_config = QConfig(
       global_config = QLayerConfig(input_tensors=Int8Spec(), weight=Int8Spec()),
       specific_layer_config: dict[QLayerConfig, list[str]] | None = None,
       layer_type_config: dict[QLayerConfig | None, list[str]] | None = None,
       exclude: list[Union[str, list[tuple[list[str]]]]] | None = None,
       algo_config: list[AlgoConfig] | None = None,
       use_external_data_format: bool = False,
       extra_options: dict[str, Any] | None = None,
    )

*  **global_config**: (QLayerConfig) The global quantization configuration applied to all layers unless overridden. Defaults to QLayerConfig(activaiton=Int8Spec(), weight=Int8Spec()) .

   -  input_tensors/weight (QTensorConfig): The Tensor-level quantization configuration of input_tensors or weight. The options are Int8Spec, UInt8Spec, XInt8Spec, Int16Spec, UInt16Spec, Int32Spec, UInt32Spec, BFloat16Spec, BFP16Spec, Int4Spec, UInt4Spec. It includes attributes whether symmetric quantization is used, the type of scaling strategy, the calibration method applied, and the level of quantization granularity.

      - symmetric (bool): Whether use symmetric quantization for QTensorConfigs like Int8Spec. For signed data types such as Int8Spec, the default value is True, while for unsigned data types such as UInt8Spec, the default value is False.
      - scale_type (ScaleType): The scale type of QTensorConfigs like Int8Spec. The options are ScaleType.Float32, ScaleType.PowerOf2 and ScaleType.Int16.
      - calibration_method (CalibMethod): The calibration method of QTensorConfigs like Int8Spec. The options are CalibMethod.MinMax, CalibMethod.MinMSE, CalibMethod.Percentile, CalibMethod.Entropy, CalibMethod.LayerwisePercentile and CalibMethod.Distribution.
      - quant_granularity (QuantGranularity): The quantization granularity of QTensorConfigs like Int8Spec. The options are QuantGranularity.Tensor, QuantGranularity.Channel and QuantGranularity.Group.

*  **specific_layer_config**: (Dictionary or None) Dictionary that maps layer identifiers to specific quantization. For example: Individual layer specification: {QLayerConfig(input_tensors=Int8Spec(), weight=Int16Spec(), bias=Int16Spec(), output_tensors=Int8Spec()): ["/conv1/Conv", "/conv2/Conv"]}; layer name pattern specification: {QLayerConfig(input_tensors=Int16Spec(), weight=Int16Spec(), bias=Int16Spec(), output_tensors=Int16Spec()): ["^/conv1/.*", "^/conv2/.*"]}; subgraph specification {QLayerConfig(input_tensors=Int16Spec(), weight=Int8Spec(), bias=Int8Spec(), output_tensors=Int16Spec()): [(["Conv1", "Conv2"], ["Relu9", "MatMul10"])]}, where the subgraph starts with "Conv1" and "Conv2" and ends with "Relu9" and "MatMul10". Defaults to None. **Note**: For example, if "/conv1/Conv"'s output tensor is "/conv2/Conv"'s input_tensor, the quantization of "/conv1/Conv"'s output tensor will be reset by "/conv2/Conv"'s input_tensor because "/conv2/Conv" is written behind by "/conv1/Conv".
*  **layer_type_config**: (Dictionary or None) Dictionary mapping all nodes to the given operation type to their specific quantizaiton like {QLayerConfig(input_tensors=Int8Spec(), weight=Int16Spec(), bias=Int16Spec(), output_tensors=Int16Spec()): ["Conv", "ConvTranspose"], None: ["MatMul", "Gemm"]}. Key is None means excluding all nodes of these operation types. Defaults to None.
*  **exclude**: (List or None) Excludes the nodes specified, nodes matched by regular expressions (Must start with ^ and contain the .* characters), and the specified subgraphs from quantization. "/conv1/Conv" is the name of a node; "^/layer0/.*" is a regular expression pattern; (["Conv1", "Conv2"], ["Relu9", "MatMul10"]) is a subgraph that starts with "Conv1" and "Conv2" and ends with "Relu9" and "MatMul10". Defaults to None.
*  **algo_config**: (List) Each element in this list is an instance of an algorithm class like [CLEConfig(cle_steps=2), AdaRoundConfig(learning_rate=0.1, num_iterations=100)]. Defaults to None.
*  **use_external_data_format**: (Boolean) This option is used for large size (>2GB) model. The model proto and data will be stored in separate files. The default is False.
*  **extra_options**: (dict[str, Any] or None) The various options for different cases. Current used:

   -  **PreprocessYAML**: (Strings or None) If specified, it will perform preprocessing before quantization according to this YAML file (e.g., "/path/to/preprocess.yaml") . Please refer to :doc:`Shapeshifter <../quark_shapeshifter>` for the YAML file. We also provide 5 common pre-defined preprocess configs (e.g., "xint8" or "a8w8" or "a16w8" or "bf16" or "bfp16"). It defaults to None.
   -  **SkipPreprocess**: (Strings or None) If True, all preprocess graph optimization like simplification, fusing operations will be skipped. If the Shapeshifter (:doc:`Shapeshifter <../quark_shapeshifter>`) is used prior to quantization, this parameter must be set to True. It defaults to False.
   -  **OpTypesToQuantize**: (List of Strings or None) If specified, only operators of the given types will be quantized (e.g., ['Conv'] to only quantize Convolutional layers). By default, all supported operators will be quantized.
   -  **NodesToQuantize**: (List of Strings or None) If specified, only given nodes will be quantized (e.g., ['/layer0/Conv_1', '/layer1/MatMul_2'] to only quantize these two nodes). Default is None.
   -  **ExtraOpTypesToQuantize**: (List of Strings or None) If specified, the given operator types will be included as additional targets for quantization, expanding the set of operators to be quantized without replacing the existing configuration (e.g., ['Gemm'] to include Gemm layers in addition to the currently specified types). By default, no extra operator types will be added for quantization.
   -  **ExecutionProviders**: (List of Strings) This parameter defines the execution providers that will be used by ONNX Runtime to do calibration for the specified model. The default value 'CPUExecutionProvider' implies that the model will be computed using the CPU as the execution provider. You can also set this to other execution providers supported by ONNX Runtime such as 'ROCMExecutionProvider' and 'CUDAExecutionProvider' for GPU-based computation, if they are available in your environment. The default is ['CPUExecutionProvider'].
   -  **OptimizeModel**:(Boolean) If True, optimizes the model before quantization. Model optimization performs certain operator fusion that makes quantization tool's job easier. For instance, a Conv/ConvTranspose/Gemm operator followed by BatchNormalization can be fused into one during the optimization, which can be quantized very efficiently. The default value is True.
   -  **ConvertFP16ToFP32**: (Boolean) This parameter controls whether to convert the input model from float16 to float32 before quantization. For float16 models, it is recommended to set this parameter to True. The default value is False. When using convert_fp16_to_fp32 in AMD Quark for ONNX, it requires onnxslim to simplify the ONNX model. Please make sure that onnxslim is installed by using 'python -m pip install onnxslim'.
   -  **ConvertNCHWToNHWC**: (Boolean) This parameter controls whether to convert the input NCHW model to input NHWC model before quantization. For input NCHW models, it is recommended to set this parameter to True. The default value is False.
   -  **DebugMode**: (Boolean) Flag to enable debug mode. In this mode, all debugging message will be printed. Default is False.
   -  **CryptoMode**: (Boolean) Flag to enable crypto mode. In this mode, all message will be blocked, and all intermediate data related to the model will not be saved to disk. In addition, the input model to the *quantize_model* API should be a ModelProto object. Please that it only supports <2GB ModelProto object. Default is False.
   -  **PrintSummary**: (Boolean) Flag to print summary of quantization. Default is True.
   -  **IgnoreWarnings**: (Boolean) Flag to suppress the warnings globally. Default is True.
   -  **ActivationScaled**: (Boolean) If True, activations will be scaled to a reduced numeric range, only configurable for floating-point quantization types, such as BFloat16. The default is False, which means by default it will cast float32 tensors to quantization types directly.
   -  **WeightScaled**: (Boolean) If True, weights will be scaled to a reduced numeric range, only configurable for floating-point quantization types, such as BFloat16. The default is False, which means by default it will cast float32 tensors to quantization types directly.
   -  **QuantizeFP16**: (Boolean) If True, the data type of the input model should be float16. It only takes effect when onnxruntime version is 1.18 or above. The default is False. When the input model is detected as FP16 and ``ConvertFP16ToFP32`` is not enabled, ``QuantizeFP16`` is auto-enabled (and ``optimize_model`` is forced to False) at runtime; see ``quark/onnx/quantization/quantize.py`` for the auto-enable logic.
   -  **UseFP32Scale**: (Boolean) If True, the scale of the quantized model is converted from float16 to float32 when the quantization is done. It only takes effect only if QuantizeFP16 is True. It must be False when UseMatMulNBits is True. The default mirrors QuantizeFP16 (effectively False unless QuantizeFP16 is True).
   -  **UseUnsignedReLU**: (Boolean) If True, the output tensor of ReLU and Clip, whose min is 0, will be forced to be asymmetric. The default is False.
   -  **QuantizeBias**: (Boolean) If True, quantize the Bias as a normal weights. The default is True. For DPU/NPU devices, this must be set to True.
   -  **Int32Bias**: (Boolean) If True, bias will be quantized in int32 data type; if false, it will have the same data type as weight. The default is False when enable_npu_cnn is True. Otherwise the default is True.
   -  **Int16Bias**: (Boolean) If True, bias will be quantized in int16 data type; The default is False. **Note**: 1. ONNXRuntime only supports Int16 Bias inference when the opset version is 21 or higher, so please ensure that the input model's opset version is 21 or higher. 2. It is recommended to use this together with ADAROUND or ADAQUANT; otherwise, the quantized model with Int16 bias may suffer from poor accuracy.
   -  **RemoveInputInit**: (Boolean) If True, initializer in graph inputs will be removed because it will not be treated as constant value/weight. This may prevent some of the graph optimizations, like const folding. The default is True.
   -  **SimplifyModel**: (Boolean) If True, The input model will be simplified using the onnxslim tool. The default is True.
   -  **SimplifyModelOptions**: (Dictionary) Optional configuration dictionary for onnxslim optimization. All parameters are passed directly to ``onnxslim.slim()`` as keyword arguments. Example configurations (e.g., {'skip_fusion_patterns': ['EliminationSlice']} to skip slice fusion, or {'model_check': True, 'skip_fusion_patterns': ['EliminationSlice'], 'save_as_external_data': False, 'verbose': True} for a complete configuration). The default is an empty dict ({}), which uses default onnxslim settings.
   -  **EnableSubgraph**: (Boolean) If True, the subgraph will be quantized. The default is False. More support for this feature is planned in the future.
   -  **EnableDualQuantNodePairs**: (Boolean) If True, there will be two pairs of quantization nodes (QuantizeLinear/DeQuantizeLinear or BFPQuantizeDequantize or MXQuantizeDequantize) at the boundary tensors of two different precision, ensuring that the node’s activation inputs and outputs have consistent QDQ types. The default is False.
   -  **ForceQuantizeNoInputCheck**: (Boolean) If True, latent operators such as maxpool and transpose will always quantize their inputs, generating quantized outputs even if their inputs have not been quantized. The default behavior can be overridden for specific nodes using nodes_to_exclude.
   -  **MatMulConstBOnly**: (Boolean) If True, only MatMul operations with a constant 'B' will be quantized. The default is False for static mode and True for dynamic mode.
   -  **AddQDQPairToWeight**: (Boolean) If True, both QuantizeLinear and DeQuantizeLinear nodes are inserted for weight, maintaining its floating-point format. The default depends on the weight dtype: forced to True for BFP/MX weight types, defaults to True for FP16/BF16 weight types, and defaults to False for integer weight types. When the default is False, floating-point weight is quantized and fed solely to an inserted DeQuantizeLinear node. In the PowerOfTwoMethod calibration method, this setting will also be effective for the bias.
   -  **OpTypesToExcludeOutputQuantization**: (List of Strings or None) If specified, the output of operators with these types will not be quantized. The default is an empty list.
   -  **NodesToExcludeOutputQuantization**: (List of Strings or None) If specified, the output of nodes with these names will not be quantized. The default is an empty list.
   -  **DedicatedQDQPair**: (Boolean) If True, an identical and dedicated QDQ pair is created for each node. The default is False, allowing multiple nodes to share a single QDQ pair as their inputs.
   -  **QDQOpTypePerChannelSupportToAxis**: (Dictionary) Sets the channel axis for specific operator types (e.g., {'MatMul': 1}). This is only effective when per-channel quantization is supported and per_channel is True. If a specific operator type supports per-channel quantization but no channel axis is explicitly specified, the default channel axis will be used. For DPU/NPU devices, this must be set to {} as per-channel quantization is currently unsupported. The default is an empty dict ({}).
   -  **CalibTensorRangeSymmetric**: (Boolean) If True, the final range of the tensor during calibration will be symmetrically set around the central point "0". The default depends on ``calibrate_method``: True for ``PowerOfTwoMethod.{NonOverflow,MinMSE}``, ``CalibrationMethod.Percentile``, and ``LayerWiseMethod.LayerWisePercentile``; False for ``CalibrationMethod.{MinMax,Entropy,Distribution}``.
   -  **CalibMovingAverage**: (Boolean) If True, the moving average of the minimum and maximum values will be computed when the calibration method selected is MinMax. The default is False. In PowerOfTwoMethod calibration method, this should be set to False.
   -  **CalibMovingAverageConstant**: (Float) Specifies the constant smoothing factor to use when computing the moving average of the minimum and maximum values. The default is 0.01. This is only effective when the calibration method selected is MinMax and CalibMovingAverage is set to True. In PowerOfTwoMethod calibration method, this option is unsupported.
   -  **Percentile**: (Float) If the calibration method is set to 'quark.onnx.CalibrationMethod.Percentile,' then this parameter can be set to the percentage for percentile. The default is 99.999.
   -  **LWPMetric**: (String) If the calibration method is set to 'quark.onnx.LayerWiseMethod.LayerWisePercentile,' then this parameter can be set to select the metric to judge the percentile value. The default is mae.
   -  **PercentileCandidates**: (List) If the calibration method is set to 'quark.onnx.LayerWiseMethod.LayerWisePercentile' then this parameter can be set to the percentage for percentiles. The default is [99.99, 99.999, 99.99999].
   -  **UseRandomData**: (Boolean) Required to be true when the RandomDataReader is needed. The default value is false.
   -  **RandomDataReaderInputShape**: (Dict) It is required to use dict {name : shape} to specify a certain input. For example, RandomDataReaderInputShape={"image" : [1, 3, 224, 224]} for the input named "image". The default value is an empty dict {}.
   -  **RandomDataReaderInputDataRange**: (Dict or None) Specifies the data range for each input when the random data reader is used (``calibration_data_reader`` is None). If set to None, every input is filled with ``numpy.random.random(input_shape)`` values in ``[0, 1)``. If a dict is supplied, it must contain an explicit ``[min, max]`` range for every input name — missing entries raise a ``ValueError``; there is no dtype-based automatic fallback. The default is None.
   -  **Int16Scale**: (Boolean) If True, the float scale will be replaced by the closest value corresponding to M and 2\ **N, where the range of M and 2**\ N is within the representation range of int16 and uint16. The default is False.
   -  **MinMSEModePof2Scale**: (String) When using quark.onnx.PowerOfTwoMethod.MinMSE, you can specify the method for calculating minmse. By default, minmse is calculated using all calibration data. Alternatively, you can set the mode to "MostCommon", where minmse is calculated for each batch separately and take the most common value. The default setting is 'All'.
   -  **NumBins**: (Int) Controls the number of histogram bins used by all histogram-based calibration methods: ``CalibrationMethod.Percentile``, ``CalibrationMethod.Entropy``, ``CalibrationMethod.Distribution``, ``LayerWiseMethod.LayerWisePercentile``, and ``PowerOfTwoMethod.MinMSE`` in ``"All"`` mode.
      For **Percentile, Entropy, Distribution, and LayerWise Percentile**, ``NumBins`` sets the resolution of the histogram that is built over the observed activation values and then used to compute the calibration threshold (e.g. a percentile cut-off or an entropy-minimising bin boundary). A higher value gives a finer histogram and a more accurate threshold, at the cost of slightly more memory and computation. The default is 2048 for Percentile, Distribution, and LayerWise Percentile; 128 for Entropy.
      For **MinMSE in "All" mode** (``MinMSEModePof2Scale="All"``), when ``NumBins > 0``, instead of accumulating every raw calibration batch on disk or in memory, each tensor's data is summarised into a fixed-size histogram that is updated online as batches arrive. The MSE for each power-of-two scale candidate is then computed from the histogram rather than from the raw values, reducing peak storage from O(N × tensor_elements) to O(NumBins) per tensor regardless of the number of calibration batches. Setting ``NumBins=0`` disables this histogram mode, restoring the original behaviour of accumulating all raw calibration data on disk or in memory (controlled by ``CalibOptimizeMem``). The default is 2048.
   -  **ConvertOpsetVersion**: (Int or None) Specifies the target opset version for the ONNX model. If set, the model's opset version will be updated accordingly. The default is None.
   -  **ConvertBNToConv**: (Boolean) If True, the BatchNormalization operation will be converted to Conv operation. The default is True when any of the following holds: ``enable_npu_cnn=True``, ``enable_npu_transformer=True``, or ``quant_format`` is ``ExtendedQuantFormat.QDQ`` without ``BF16QDQToCast``/``EnableVaimlBF16`` enabled. Otherwise the default is False.
   -  **ConvertReduceMeanToGlobalAvgPool**: (Boolean) If True, the Reduce Mean operation will be converted to Global Average Pooling operation. Defaults follow the same condition as ``ConvertBNToConv``.
   -  **SplitLargeKernelPool**: (Boolean) If True, the large kernel Global Average Pooling operation will be split into multiple Average Pooling operation. Defaults follow the same condition as ``ConvertBNToConv``.
   -  **ConvertSplitToSlice**: (Boolean) If True, the Split operation will be converted to Slice operation. Defaults follow the same condition as ``ConvertBNToConv``.
   -  **FuseInstanceNorm**: (Boolean) If True, the split instance norm operation will be fused to InstanceNorm operation. The default is True.
   -  **FuseL2Norm**: (Boolean) If True, a set of L2norm operations will be fused to L2Norm operation. The default is True.
   -  **FuseGelu**: (Boolean) If True, a set of Gelu operations will be fused to Gelu operation. The default is True.
   -  **FuseLayerNorm**: (Boolean) If True, a set of LayerNorm operations will be fused to LayerNorm operation. The default is True.
   -  **ConvertClipToRelu**: (Boolean) If True, the Clip operations that has a min value of 0 will be converted to ReLU operations. The default is False. Built-in preset configs (e.g. VINT8_CONFIG) may set this to True explicitly via extra_options.
   -  **SimulateDPU**: (Boolean) If True, a simulation transformation that replaces some operations with an approximate implementation will be applied. This is consumed both by the XINT8 NPU CNN path (when ``enable_npu_cnn`` is True) and by the Extended QDQ path. The default is True.
   -  **ConvertLeakyReluToDPUVersion**: (Boolean) If True, the Leaky Relu operation will be converted to DPU version when SimulateDPU is True. The default is True.
   -  **ConvertSigmoidToHardSigmoid**: (Boolean) If True, the Sigmoid operation will be converted to Hard Sigmoid operation when SimulateDPU is True. The default is True.
   -  **ConvertHardSigmoidToDPUVersion**: (Boolean) If True, the Hard Sigmoid operation will be converted to DPU version when SimulateDPU is True. The default is True.
   -  **ConvertAvgPoolToDPUVersion**: (Boolean) If True, the global or kernel-based Average Pooling operation will be converted to DPU version when SimulateDPU is True. The default is True.
   -  **ConvertClipToDPUVersion**: (Boolean) If True, the Clip operation will be converted to DPU version when SimulateDPU is True. The default is False.
   -  **ConvertReduceMeanToDPUVersion**: (Boolean) If True, the ReduceMean operation will be converted to DPU version when SimulateDPU is True. The default is True.
   -  **ConvertSoftmaxToDPUVersion**: (Boolean) If True, the Softmax operation will be converted to DPU version when SimulateDPU is True. The default is False.
   -  **NPULimitationCheck**: (Boolean) If True, the quantization position will be adjust due to the limitation of DPU/NPU. The default is True.
   -  **MaxLoopNum**: (Int) The quantizer adjusts or aligns the quantization position through loops, this option is used to set the maximum number of loops. The default value is 5.
   -  **AdjustShiftCut**: (Boolean) If True, adjust the shift cut of nodes when NPULimitationCheck is True. The default is True.
   -  **AdjustShiftBias**: (Boolean) If True, adjust the shift bias of nodes when NPULimitationCheck is True. The default is True.
   -  **AdjustShiftRead**: (Boolean) If True, adjust the shift read of nodes when NPULimitationCheck is True. The default is True.
   -  **AdjustShiftWrite**: (Boolean) If True, adjust the shift write of nodes when NPULimitationCheck is True. The default is True.
   -  **AdjustHardSigmoid**: (Boolean) If True, adjust the position of hard sigmoid nodes when NPULimitationCheck is True. The default is True.
   -  **AdjustShiftSwish**: (Boolean) If True, adjust the shift swish when NPULimitationCheck is True. The default is True.
   -  **AlignConcat**: (Boolean) If True, adjust the quantization position of concat when NPULimitationCheck is True. The default is True, when the power-of-two scale is used, otherwise it's False.
   -  **AlignPool**: (Boolean) If True, adjust the quantization position of pooling when NPULimitationCheck is True. The default is True, when the power-of-two scale is used, otherwise it's False.
   -  **AlignPad**: (Boolean) If True, adjust the quantization position of pad when NPULimitationCheck is True. The default is True, when the power-of-two scale is used, otherwise it's False.
   -  **AlignSlice**: (Boolean) If True, adjust the quantization position of slice when NPULimitationCheck is True. The default is True, when the power-of-two scale is used, otherwise it's False.
   -  **AlignTranspose**: (Boolean) If True, adjust the quantization position of transpose when NPULimitationCheck is True. The default is False.
   -  **AlignReshape**: (Boolean) If True, adjust the quantization position of reshape when NPULimitationCheck is True. The default is False.
   -  **AdjustBiasScale**: (Boolean) If True, adjust the bias scale equal to input_tensors scale multiply by weights scale. The default is True.
   -  **SaveAndRestore**: (None or String) This parameter is used to control saving and restoring of calibration and fine-tuning data, and it should be a ".json" file. If set to None, the calibration and fine-tuning start from scratch, and no data is saved. If set to a file path and the file does not exist, the calibration tensors' range, the fine-tuning intermediate quantized model and the indices of layers selected for fine-tuning are computed and saved to that file. If the file already exists, the saved calibration tensors' range, intermediate quantized model and layers' indices are loaded and fine-tuning continues (provided fine-tuning is enabled and the selected layer indices are within the allowed constraints). If fine-tuning is not enabled, SaveAndRestore is used only for saving and restoring calibration information. This file enables saving, restoring, and controlling calibration data and selectively fine-tuning specific layers. The default value is None.
   -  **ReplaceClip6Relu**: (Boolean) If True, Replace Clip(0,6) with Relu in the model. This option is only available when CLE algorithm is enabled. The default is False.
   -  **CopySharedInit**: (List or None) Specifies the node op_types to run duplicating initializer in the model for separate quantization use across different nodes, e.g. ['Conv', 'Gemm', 'Mul'] input, only shared initializer in these nodes will be duplicated. None means that skip this conversion while empty list means that run this for all op_types included in the given model, default is None.
   -  **CopyBiasInit**: (List or None) Specifies the node operation types to run duplicating bias initializer in the model for separate quantization use across different nodes, e.g. ['Conv', 'Gemm', 'Mul'] input, only shared bias initializer in these nodes will be duplicated. None means that skip this conversion while empty list means that run this for all operation types included in the given model. The default is ``["Conv", "ConvTranspose", "Gemm"]``. The pass is only executed when both activation and weight types are int8/int16 (e.g. A8W8, A16W8); for other quantization configurations the pass is skipped regardless of this value.
   -  **RemoveQDQConvClip**: (Boolean) If True, the QDQ between Conv/Add/Gemm and Clip will be removed for DPU. The default is True.
   -  **RemoveQDQConvRelu**: (Boolean) If True, the QDQ between Conv/Add/Gemm and Relu will be removed for DPU. The default is True.
   -  **RemoveQDQConvLeakyRelu**: (Boolean) If True, the QDQ between Conv/Add/Gemm and LeakyRelu will be removed for DPU. The default is True.
   -  **RemoveQDQConvPRelu**: (Boolean) If True, the QDQ between Conv/Add/Gemm and PRelu will be removed for DPU. The default is True.
   -  **RemoveQDQConvGelu**: (Boolean) If True, the QDQ between Conv/Add/Gemm and Gelu will be removed. The default is False.
   -  **RemoveQDQMulAdd**: (Boolean) If True, the QDQ between Mul and Add will be removed for NPU. The default is False.
   -  **RemoveQDQBetweenOps**: (List of tuples (Strings, Strings) or None) This parameter accepts a list of tuples representing operation type pairs (e.g., Conv and Relu). If set, the QDQ between the specified pairs of operations will be removed for NPU. The default is None.
   -  **RemoveQDQInstanceNorm**: (Boolean) If True, the QDQ between InstanceNorm and Relu/LeakyRelu/PRelu will be removed for DPU. The default is False.
   -  **RemoveFusedQDQ**: (Boolean) If True, the fused QDQ nodes, such as BFPQuantizeDequantize and MXQuantizeDequantize, will be removed as well when removing QDQ nodes options are enabled. The default is False.
   -  **FoldBatchNorm**: (Boolean) If True, the BatchNormalization operation will be fused with Conv, ConvTranspose or Gemm operation. The BatchNormalization operation after Concat operation will also be fused, if the all input operations of the Concat operation are Conv, ConvTranspose or Gemm operations. The default mirrors the effective ``optimize_model`` flag (True when ``optimize_model`` is enabled, False otherwise — note that several pipelines, e.g. FP16/crypto, force ``optimize_model=False`` internally and therefore also disable ``FoldBatchNorm`` by default).
   -  **BF16WithClip**: (Boolean) If True, during BFloat16 quantization, insert "Clip" node before customized "QuantizeLinear" node to add boundary protection for input_tensors. The default is False.
   -  **BF16QDQToCast**: (Boolean) If True, during BFloat16 quantization, replace QuantizeLinear/DeQuantizeLinear ops with Cast ops to accelerate BFloat16 quantized inference. The default is False.
   -  **RefineBlockAxis**: (Boolean) If True, the block axes of BFP or MX quantization nodes will be refined. The axis will always be on the K (reduction) dimensions for matrix multiplication operations (e.g. Gemm and MatMul). For Softmax operations, the axis should be consistent with the axis attribute of the Softmax operation. When the quantization type of activation or weight is BFP or MX, the default is True. Otherwise it's False.
   -  **FixShapes**: (String) Set the input and output shapes of the quantized model to a fixed shape by default if not explicitly specified. The example: 'FixShapes':'input_1:[1,224,224,3];input_2:[1,96,96,3];output_1:[1,100];output_2:[1,1000]'
   -  **FoldRelu**: (Boolean) If True, the Relu will be fold to Conv when use ExtendedQuantFormat. The default is False.
   -  **CalibDataSize**: (Int) This parameter controls how many data are used for calibration. The default to using all the data in the calibration dataloader.
   -  **CalibWorkerNum**: (Int) This parameter controls how many workers (processes) to collect data. The more workers there are, the less time it takes, but the more memory it consumes (because each worker requires independent memory space). It supports all methods except for CalibrationMethod.MinMax and PowerOfTwoMethod.NonOverflow. The default is 1.
   -  **CalibOptimizeMem**: (Boolean) If set to True, most calibration methods will cache intermediate data of activation tensors on disk to reduce memory consumption. However, for MinMax and NonOverflow methods, it disables the memory pool technology during model inference to prevent additional memory usage. The default setting is True for every ``calibrate_method`` except ``LayerWiseMethod.LayerWisePercentile``, where the default is False; and when both ``CalibOptimizeDisk`` and ``CalibOptimizeMem`` are True under ``LayerWisePercentile``, ``CalibOptimizeMem`` is force-disabled (the two are mutually exclusive).
   -  **CalibOptimizeDisk**: (Boolean) If set to True, `LayerwisePercentile` calibration method avoids caching intermediate activation tensors—neither writing them to disk nor retaining them in memory. Instead, each sample is processed on the fly and immediately discarded, and recomputed only if needed. This significantly reduces memory usage to a minimum and eliminates disk consumption entirely, at the cost of increased runtime (typically less than 2 times). The default value is True.
   -  **SaveTensorHistFig**: (Boolean) If True, save the tensor histogram to the file 'tensor_hist' in the working directory during calibration. Only available for histogram-based calibration methods, such as Percentile, Entropy and Distribution. The default is False.
   -  **QuantizeAllOpTypes**: (Boolean) If True, all operation types will be quantized. The default is False. Built-in preset configs (BF16_CONFIG, BF16_ADAQUANT_CONFIG, VINT8_CONFIG) set this to True explicitly via extra_options.
   -  **WeightsOnly**: (Boolean) If True, only quantize weights of the model. The default is False.
   -  **AlignEltwiseQuantType**: (Boolean) If True, quantize weights of the node with the input_tensors quant type if node type in [Mul, Add, Sub, Div, Min, Max] when quant_format is ExtendedQuantFormat.QDQ and enable_npu_cnn is False and enable_npu_transformer is False. The default is False.
   -  **EnableVaimlBF16**: (Boolean) If True, the bfloat16 quantized model with vitis qdq will be converted to a bfloat16 quantized model with bfloat16 weights stored as float32. Vaiml is the name of a compiler, the bfloat16 quantized model can be directly deployed on the compiler if the parameter is True. The default is False.
   -  **UseMatMulNBits**: (Boolean) If True, only quantize weights with nbits for MatMul of the model. The default is False.
   -  **MatMulNBitsParams**: (Dictionary) A parameter used to specify the settings for MatMulNBits Quantizer:

      -  **Algorithm**: (str) The algorithm in MatMulNBits Quantization determines which algorithm ("DEFAULT", "GPTQ", "HQQ") to be used to quantize weights. The default is "DEFAULT".
      -  **GroupSize**: (int) The block size in MatMulNBits Quantization determines how many weights share a scale. The default is 128.
      -  **Symmetric**: (Boolean) If True, symmetrize quantization for weights. The default is True.
      -  **Bits**: (int) The target bits to quantize. The default is 4. Only 4b quantization is supported for inference, additional bits support is planned.
      -  **AccuracyLevel**: (int) The quantization level of input, can be: 0(unset), 1(fp32), 2(fp16), 3(bf16), or 4(int8). The default is 0.

   -  **EvalMetrics**: (Boolean) If True, enables evaluation of the quantized model by measuring cosine similarity and L2 loss. The default is False.
   -  **EvalDataReader**: (DataReader) This parameter is used only when EvalMetrics is set to True. It allows the user to provide a custom data reader for evaluating the quantized model's cosine similarity and L2 loss metrics against the float model.
   -  **TmpDir**: (String) Specifies the directory used to cache intermediate files. The default value is None, in which case the system temporary directory will be used for the caching. This argument can be set for either absolute or relative path.
   -  **UserCustomOpLibPath**: (String) Specifies the directory for user-defined custom operation libraries (in both Python and C++ implementations) that define custom operations (custom ops), enabling support for floating-point models with custom ops during quantization. The default value is None. Accepts both absolute and relative paths to the .so file directory.
   -  **EncryptionAlgorithm**: (String) A parameter used to specify the encryption algorithm for crypto mode, only "AES-256" algorithm is supported currently. The default value is None, which means it will not save any intermediate models/files to disk in crypto mode.
   -  **WeightCalibrateMethod**: (String) Specifies the weight calibration method.
      Options: 'quark.onnx.CalibrationMethod.MinMax' – Use tensor minimum and maximum values,
      'quark.onnx.ExtendedCalibrationMethod.MinMSE' – Search for a float scale that minimizes MSE (Mean Squared Error).
      Default: None – It behaves as 'quark.onnx.CalibrationMethod.MinMax'.
   -  **MinMSEModeFloatScale**: (String) Only effective when 'WeightCalibrateMethod' is quark.onnx.ExtendedCalibrationMethod.MinMSE.
      Options: "Percentile" - Select the percentile from 99.9, 99.99, 99.999, or 99.9999 that gives the minimal MSE,
      "HistCenter" (use ≤2048-bin histogram centers to minimize weighted MSE),
      "All" (use full data to minimize MSE).
      Unsupported options fall back to "Percentile". The default is "Percentile". (Note: options are case-sensitive.)
   -  **TensorQuantOverrides**: (Dictionary[String, List[Dictionary[String, Any]]]) Set tensor quantization overrides, the default is {}. The key is a tensor name and the value is a list of dictionaries. For per-tensor quantization, the list contains a single dictionary. For per-channel quantization, the list contains a dictionary for each channel in the tensor. Each dictionary in the list contains optional overrides with the following keys and values. For example, this configuration extra_options["TensorQuantOverrides"] = { "/model/conv_1/weight": [ {"quant_type": QuantType.Int8, "axis": 1 }], "/model/conv_2/output_0": [ { "quant_type": ExtendedQuantType.QBFP } ] } overrides a conv node's weight with Int8 per-channel quantization and another conv node's output with BFP data type.

      -  **quant_type**: (QuantType | ExtendedQuantType) The tensor's quantization data type.
      -  **axis**: (Int) The axis to perform per-channel quantization. Only available for integer data types.
      -  **scale**: (Float) The scale value to use. Must also specify `zero_point` if set.
      -  **zero_point**: (Int) The zero-point value to use. Must also specify `scale` is set.
      -  **symmetric**: (Bool) If the tensor should use symmetric quantization. Invalid if also set `scale` or `zero_point`.
      -  **reduce_range**: (Bool) If the quantization range should be reduced. Invalid if also set `scale` or `zero_point`.
      -  **rmax**: (Float) Override the maximum real tensor value in calibration data. Invalid if also set `scale` or `zero_point`.
      -  **rmin**: (Float) Override the minimum real tensor value in calibration data. Invalid if also set `scale` or `zero_point`.
   - **CalibPassthroughOpTypes**: (List of str) defines the set of ONNX operator types that are considered distribution-preserving during Selective Calibration Propagation (SCP). Operators listed in this parameter do not require independent calibration because they do not alter the statistical distribution of tensor values. Recommended setting is ["Reshape", "Transpose", "MaxPool", "Split", "Slice", "Squeeze", "Unsqueeze", "Gather",], default is [].
   -  **FillAllValueInfo**: (Boolean) If True, value info (data type and shape) is filled for all intermediate tensors in the quantized model after quantization. This is required by many downstream compilers to infer tensor shapes throughout the graph. The default is False.

Table 7. Quantization Data Types can be selected

+----------------------------------+---------------------------+
| data_type                        | comments                  |
+==================================+===========================+
| Int4                             |                           |
| UInt4                            |                           |
| Int8                             |                           |
| UInt8                            |                           |
| Int16                            |                           |
| UInt16                           |                           |
| Int32                            |                           |
| UInt32                           |                           |
| BFloat16                         |                           |
| BFP16                            |                           |
+----------------------------------+---------------------------+
