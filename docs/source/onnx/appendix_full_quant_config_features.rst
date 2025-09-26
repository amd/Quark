.. Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.

Full List of Quantization Configuration Features
================================================

Overview
--------

It's very simple to quantize a model using the ONNX quantizer of Quark, only a few straightforward Python statements:

.. code:: python

    from quark.onnx import ModelQuantizer
    from quark.onnx.quantization import QConfig
    from quark.onnx.quantization.config.spec import QLayerConfig, Int8Spec

    config = QConfig(global_config=QLayerConfig(activation=Int8Spec(), weight=Int8Spec()))
    quantizer = ModelQuantizer(config)
    quantizer.quantize_model(model_input, model_output, calibration_data_reader)


As shown in the code, just create a quantization configuration and use it to initialize a quantizer, and then call the quantizer's *quantize_model()* API, which has 3 main parameters:

*  **model_input**: (String or ModelProto) This parameter specifies the file path of the model that is to be quantized. When a file path cannot be specified, the loaded ModelProto can also be passed in directly.
*  **model_output**: (Optional String) This parameter specifies the file path where the quantized model will be saved. You can leave it unspecified (it will default to None), and the ModelProto format quantized model will be returned by the API.
*  **calibration_data_reader**: (Optional Object) This parameter is a calibration data reader that enumerates the calibration data and generates inputs for the original model. You can leave it unspecified (it will default to None), and simply enable *UseRandomData* in extra options of quantization configuration to use random data for calibration.

The next section will provide a detailed list of all parameters in the quantization configuration.

Quantization Configuration
--------------------------

.. code:: python

    from quark.onnx.quantization.spec import QLayerConfig, Int8Spec
    from quark.onnx.quantization import QConfig

    quant_config = QConfig(
       global_config = QLayerConfig(activation=Int8Spec(), weight=Int8Spec()),
       specific_layer_config: dict[DataType, list[str]] | None = None,
       layer_type_config: dict[DataType | None, list[str]] | None = None,
       exclude: list[Union[str, list[tuple[list[str]]]]] | None = None,
       algo_config: list[AlgoConfig] | None = None,
       use_external_data_format: bool = False,
       **kwargs: Any = None
    )

*  **global_config**: (QLayerConfig) The global quantization configuration applied to all layers unless overridden. Defaults to QLayerConfig(activaiton=Int8Spec(), weight=Int8Spec()) .

   -  activation/weight (QTensorConfig): The Tensor-level quantization configuration of activation or weight. The options are Int8Spec, UInt8Spec, XInt8Spec, Int16Spec, UInt16Spec, Int32Spec, UInt32Spec, BFloat16Spec, BFP16Spec, Int4Spec, UInt4Spec. It includes attributes whether symmetric quantization is used, the type of scaling strategy, the calibration method applied, and the level of quantization granularity.

      - symmetric (bool): Whether use symmetric quantization for QTensorConfigs like Int8Spec. For signed data types such as Int8Spec, the default value is True, while for unsigned data types such as UInt8Spec, the default value is False.
      - scale_type (ScaleType): The scale type of QTensorConfigs like Int8Spec. The options are ScaleType.Float32, ScaleType.PowerOf2 and ScaleType.Int16.
      - calibration_method (CalibMethod): The calibration method of QTensorConfigs like Int8Spec. The options are CalibMethod.MinMax, CalibMethod.MinMSE, CalibMethod.Percentile, CalibMethod.Entropy, CalibMethod.LayerwisePercentile and CalibMethod.Distribution.
      - quant_granularity (QuantGranularity): The quantization granularity of QTensorConfigs like Int8Spec. The options are QuantGranularity.Tensor, QuantGranularity.Channel and QuantGranularity.Group.

*  **specific_layer_config**: (Dictionary or None) Dictionary mapping specific layer names to their quantizaiton data type like {Int16: ["/conv1/Conv", "/conv2/Conv"], BFloat16: ["/matmul1/MatMul", "/matmul2/MatMul"]}. Defaults to None.
*  **layer_type_config**: (Dictionary or None) Dictionary mapping all nodes to the given operation type to their quantizaiton data type like {Int16: ["Conv", "ConvTranspose"], None: ["MatMul", "Gemm"]}. Data type is None means excluding all nodes of these operation types. Defaults to None.
*  **exclude**: (List or None) Excludes the nodes specified, nodes matched by regular expressions, and the specified subgraphs from quantization like ["/conv1/Conv", "^/layer0/.*", (["Conv1"], ["Conv2"]), (["Relu9", "MatMul10"])]. "/conv1/Conv" is the name of a node; "^/layer0/.*" is a regular expression pattern; (["Conv1"], ["Conv2"]), (["Relu9", "MatMul10"]) is a subgraph that starts with "Conv1" and "Conv2" and ends with "Relu9" and "MatMul10". Defaults to None.
*  **algo_config**: (List) Each element in this list is an instance of an algorithm class like [CLEConfig(cle_steps=2), AdaRoundConfig(learning_rate=0.1, num_iterations=100)]. Defaults to None.
*  **use_external_data_format**: (Boolean) This option is used for large size (>2GB) model. The model proto and data will be stored in separate files. The default is False.
*  **kwargs**: (Any or None) The kwargs for various options in different cases. Current used:

   -  **OpTypesToQuantize**: (List of Strings or None) If specified, only operators of the given types will be quantized (e.g., ['Conv'] to only quantize Convolutional layers). By default, all supported operators will be quantized.
   -  **ExtraOpTypesToQuantize**: (List of Strings or None) If specified, the given operator types will be included as additional targets for quantization, expanding the set of operators to be quantized without replacing the existing configuration (e.g., ['Gemm'] to include Gemm layers in addition to the currently specified types). By default, no extra operator types will be added for quantization.
   -  **ExecutionProviders**: (List of Strings) This parameter defines the execution providers that will be used by ONNX Runtime to do calibration for the specified model. The default value 'CPUExecutionProvider' implies that the model will be computed using the CPU as the execution provider. You can also set this to other execution providers supported by ONNX Runtime such as 'ROCMExecutionProvider' and 'CUDAExecutionProvider' for GPU-based computation, if they are available in your environment. The default is ['CPUExecutionProvider'].
   -  **OptimizeModel**:(Boolean) If True, optimizes the model before quantization. Model optimization performs certain operator fusion that makes quantization tool's job easier. For instance, a Conv/ConvTranspose/Gemm operator followed by BatchNormalization can be fused into one during the optimization, which can be quantized very efficiently. The default value is True.
   -  **ConvertFP16ToFP32**: (Boolean) This parameter controls whether to convert the input model from float16 to float32 before quantization. For float16 models, it is recommended to set this parameter to True. The default value is False. When using convert_fp16_to_fp32 in AMD Quark for ONNX, it requires onnxsim to simplify the ONNX model. Please make sure that onnxsim is installed by using 'python -m pip install onnxsim'.
   -  **ConvertNCHWToNHWC**: (Boolean) This parameter controls whether to convert the input NCHW model to input NHWC model before quantization. For input NCHW models, it is recommended to set this parameter to True. The default value is False.
   -  **DebugMode**: (Boolean) Flag to enable debug mode. In this mode, all debugging message will be printed. Default is False.
   -  **CryptoMode**: (Boolean) Flag to enable crypto mode. In this mode, all message will be blocked, and all intermediate data related to the model will not be saved to disk. In addition, the input model to the *quantize_model* API should be a ModelProto object. Please that it only supports <2GB ModelProto object. Default is False.
   -  **PrintSummary**: (Boolean) Flag to print summary of quantization. Default is True.
   -  **IgnoreWarnings**: (Boolean) Flag to suppress the warnings globally. Default is True.
   -  **LogSeverityLevel**: (Int) This parameter is used to select the severity level of screen printing logs. Its value ranges from 0 to 4: 0 for DEBUG, 1 for INFO, 2 for WARNING, 3 for ERROR and 4 for CRITICAL or FATAL. Default value is 1, which means printing all messages including INFO, WARNING, ERROR and etc by default.
   -  **ActivationScaled**: (Boolean) If True, all activations will be scaled to the exact numeric range. The default is True for integer data type quantization and False for BFloat16 and Float16, which means by default the BFloat16/Float16 quantization will cast float32 tensors to BFloat16/Float16 directly.
   -  **WeightScaled**: (Boolean) If True, all weights will be scaled to the exact numeric range. The default is True for integer data type quantization and False for BFloat16 and Float16, which means by default the BFloat16/Float16 quantization will cast float32 tensors to BFloat16/Float16 directly.
   -  **QuantizeFP16**: (Boolean) If True, the data type of the input model should be float16. It only takes effect when onnxruntime version is 1.18 or above. The default is False.
   -  **UseFP32Scale**: (Boolean) If True, the scale of the quantized model is converted from float16 to float32 when the quantization is done. It only takes effect only if QuantizeFP16 is True. It must be False when UseMatMulNBits is True. The default is True.
   -  **UseUnsignedReLU**: (Boolean) If True, the output tensor of ReLU and Clip, whose min is 0, will be forced to be asymmetric. The default is False.
   -  **QuantizeBias**: (Boolean) If True, quantize the Bias as a normal weights. The default is True. For DPU/NPU devices, this must be set to True.
   -  **Int32Bias**: (Boolean) If True, bias will be quantized in int32 data type; if false, it will have the same data type as weight. The default is False when enable_npu_cnn is True. Otherwise the default is True.
   -  **Int16Bias**: (Boolean) If True, bias will be quantized in int16 data type; The default is False. **Note**: 1. ONNXRuntime only supports Int16 Bias inference when the opset version is 21 or higher, so please ensure that the input model's opset version is 21 or higher. 2. It is recommended to use this together with ADAROUND or ADAQUANT; otherwise, the quantized model with Int16 bias may suffer from poor accuracy.
   -  **RemoveInputInit**: (Boolean) If True, initializer in graph inputs will be removed because it will not be treated as constant value/weight. This may prevent some of the graph optimizations, like const folding. The default is True.
   -  **SimplifyModel**: (Boolean) If True, The input model will be simplified using the onnxsim tool. The default is True.
   -  **EnableSubgraph**: (Boolean) If True, the subgraph will be quantized. The default is False. More support for this feature is planned in the future.
   -  **ForceQuantizeNoInputCheck**: (Boolean) If True, latent operators such as maxpool and transpose will always quantize their inputs, generating quantized outputs even if their inputs have not been quantized. The default behavior can be overridden for specific nodes using nodes_to_exclude.
   -  **MatMulConstBOnly**: (Boolean) If True, only MatMul operations with a constant 'B' will be quantized. The default is False for static mode and True for dynmaic mode.
   -  **AddQDQPairToWeight**: (Boolean) If True, both QuantizeLinear and DeQuantizeLinear nodes are inserted for weight, maintaining its floating-point format. The default is False, which quantizes floating-point weight and feeds it solely to an inserted DeQuantizeLinear node. In the PowerOfTwoMethod calibration method, this setting will also be effective for the bias.
   -  **OpTypesToExcludeOutputQuantization**: (List of Strings or None) If specified, the output of operators with these types will not be quantized. The default is an empty list.
   -  **DedicatedQDQPair**: (Boolean) If True, an identical and dedicated QDQ pair is created for each node. The default is False, allowing multiple nodes to share a single QDQ pair as their inputs.
   -  **QDQOpTypePerChannelSupportToAxis**: (Dictionary) Sets the channel axis for specific operator types (e.g., {'MatMul': 1}). This is only effective when per-channel quantization is supported and per_channel is True. If a specific operator type supports per-channel quantization but no channel axis is explicitly specified, the default channel axis will be used. For DPU/NPU devices, this must be set to {} as per-channel quantization is currently unsupported. The default is an empty dict ({}).
   -  **CalibTensorRangeSymmetric**: (Boolean) If True, the final range of the tensor during calibration will be symmetrically set around the central point "0". The default is False. In PowerOfTwoMethod calibration method, the default is True.
   -  **CalibMovingAverage**: (Boolean) If True, the moving average of the minimum and maximum values will be computed when the calibration method selected is MinMax. The default is False. In PowerOfTwoMethod calibration method, this should be set to False.
   -  **CalibMovingAverageConstant**: (Float) Specifies the constant smoothing factor to use when computing the moving average of the minimum and maximum values. The default is 0.01. This is only effective when the calibration method selected is MinMax and CalibMovingAverage is set to True. In PowerOfTwoMethod calibration method, this option is unsupported.
   -  **Percentile**: (Float) If the calibration method is set to 'quark.onnx.CalibrationMethod.Percentile,' then this parameter can be set to the percentage for percentile. The default is 99.999.
   -  **LWPMetric**: (String) If the calibration method is set to 'quark.onnx.LayerWiseMethod.LayerWisePercentile,' then this parameter can be set to select the metric to judge the percentile value. The default is mae.
   -  **ActivationBitWidth**: (Int) If the calibration method is set to 'quark.onnx.LayerWiseMethod.LayerWisePercentile', then this parameter can be set to calculate the quantize/dequantize error. The default is 8.
   -  **PercentileCandidates**: (List) If the calibration method is set to 'quark.onnx.LayerWiseMethod.LayerWisePercentile' then this parameter can be set to the percentage for percentiles. The default is [99.99, 99.999, 99.9999].
   -  **UseRandomData**: (Boolean) Required to be true when the RandomDataReader is needed. The default value is false.
   -  **RandomDataReaderInputShape**: (Dict) It is required to use dict {name : shape} to specify a certain input. For example, RandomDataReaderInputShape={"image" : [1, 3, 224, 224]} for the input named "image". The default value is an empty dict {}.
   -  **RandomDataReaderInputDataRange**: (Dict or None) Specifies the data range for each inputs if used random data reader (calibration_data_reader is None). Currently, if set to None then the random value will be 0 or 1 for all inputs, otherwise range [-128,127] for unsigned int, range [0,255] for signed int and range [0,1] for other float inputs. The default is None.
   -  **Int16Scale**: (Boolean) If True, the float scale will be replaced by the closest value corresponding to M and 2\ **N, where the range of M and 2**\ N is within the representation range of int16 and uint16. The default is False.
   -  **MinMSEModePof2Scale**: (String) When using quark.onnx.PowerOfTwoMethod.MinMSE, you can specify the method for calculating minmse. By default, minmse is calculated using all calibration data. Alternatively, you can set the mode to "MostCommon", where minmse is calculated for each batch separately and take the most common value. The default setting is 'All'.
   -  **ConvertOpsetVersion**: (Int or None) Specifies the target opset version for the ONNX model. If set, the model's opset version will be updated accordingly. The default is None.
   -  **ConvertBNToConv**: (Boolean) If True, the BatchNormalization operation will be converted to Conv operation. The default is True when enable_npu_cnn is True.
   -  **ConvertReduceMeanToGlobalAvgPool**: (Boolean) If True, the Reduce Mean operation will be converted to Global Average Pooling operation. The default is True when enable_npu_cnn is True.
   -  **SplitLargeKernelPool**: (Boolean) If True, the large kernel Global Average Pooling operation will be split into multiple Average Pooling operation. The default is True when enable_npu_cnn is True.
   -  **ConvertSplitToSlice**: (Boolean) If True, the Split operation will be converted to Slice operation. The default is True when enable_npu_cnn is True.
   -  **FuseInstanceNorm**: (Boolean) If True, the split instance norm operation will be fused to InstanceNorm operation. The default is True.
   -  **FuseL2Norm**: (Boolean) If True, a set of L2norm operations will be fused to L2Norm operation. The default is True.
   -  **FuseGelu**: (Boolean) If True, a set of Gelu operations will be fused to Gelu operation. The default is True.
   -  **FuseLayerNorm**: (Boolean) If True, a set of LayerNorm operations will be fused to LayerNorm operation. The default is True.
   -  **ConvertClipToRelu**: (Boolean) If True, the Clip operations that has a min value of 0 will be converted to ReLU operations. The default is True when enable_npu_cnn is True.
   -  **SimulateDPU**: (Boolean) If True, a simulation transformation that replaces some operations with an approximate implementation will be applied for DPU when enable_npu_cnn is True. The default is True.
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
   -  **AdjustBiasScale**: (Boolean) If True, adjust the bias scale equal to activation scale multiply by weights scale. The default is True.
   -  **TensorsRangeFile**: (None or String) This parameter is used to manage tensor range information, and it should has a ".json" suffix because this file will be save in that format. When set to None, the tensor ranges will be calculated from scratch and will not be saved. If set to a string representing a file path, and the file does not exist, the tensor range information will be computed and saved to that file. If the file already exists, the tensor range information will be loaded from it and will not be recalculated. This file can help to save the calibration time when to rerun FastFinetune algorithm or to reproduce some calibrated models. The default value is None.
   -  **ReplaceClip6Relu**: (Boolean) If True, Replace Clip(0,6) with Relu in the model. The default is False.
   -  **CopySharedInit**: (List or None) Specifies the node op_types to run duplicating initializer in the model for separate quantization use across different nodes, e.g. ['Conv', 'Gemm', 'Mul'] input, only shared initializer in these nodes will be duplicated. None means that skip this conversion while empty list means that run this for all op_types included in the given model, default is None.
   -  **CopyBiasInit**: (List or None) Specifies the node operation types to run duplicating bias initializer in the model for separate quantization use across different nodes, e.g. ['Conv', 'Gemm', 'Mul'] input, only shared bias initializer in these nodes will be duplicated. None means that skip this conversion while empty list means that run this for all operation types included in the given model. The default is an empty list when using quantization with float scale like A8W8 and A16W8. The default is None otherwise.
   -  **RemoveQDQConvClip**: (Boolean) If True, the QDQ between Conv/Add/Gemm and Clip will be removed for DPU. The default is True.
   -  **RemoveQDQConvRelu**: (Boolean) If True, the QDQ between Conv/Add/Gemm and Relu will be removed for DPU. The default is True.
   -  **RemoveQDQConvLeakyRelu**: (Boolean) If True, the QDQ between Conv/Add/Gemm and LeakyRelu will be removed for DPU. The default is True.
   -  **RemoveQDQConvPRelu**: (Boolean) If True, the QDQ between Conv/Add/Gemm and PRelu will be removed for DPU. The default is True.
   -  **RemoveQDQConvGelu**: (Boolean) If True, the QDQ between Conv/Add/Gemm and Gelu will be removed. The default is False.
   -  **RemoveQDQMulAdd**: (Boolean) If True, the QDQ between Mul and Add will be removed for NPU. The default is False.
   -  **RemoveQDQBetweenOps**: (List of tuples (Strings, Strings) or None) This parameter accepts a list of tuples representing operation type pairs (e.g., Conv and Relu). If set, the QDQ between the specified pairs of operations will be removed for NPU. The default is None.
   -  **RemoveQDQInstanceNorm**: (Boolean) If True, the QDQ between InstanceNorm and Relu/LeakyRelu/PRelu will be removed for DPU. The default is False.
   -  **FoldBatchNorm**: (Boolean) If True, the BatchNormalization operation will be fused with Conv, ConvTranspose or Gemm operation. The BatchNormalization operation after Concat operation will also be fused, if the all input operations of the Concat operation are Conv, ConvTranspose or Gemm operatons.The default is True.
   -  **BF16WithClip**: (Boolean) If True, during BFloat16 quantization, insert "Clip" node before customized "QuantizeLinear" node to add boundary protection for activation. The default is False.
   -  **BF16QDQToCast**: (Boolean) If True, during BFloat16 quantization, replace QuantizeLinear/DeQuantizeLinear ops with Cast ops to accelerate BFloat16 quantized inference. The default is False.
   -  **FixShapes**: (String) Set the input and output shapes of the quantized model to a fixed shape by default if not explicitly specified. The example: 'FixShapes':'input_1:[1,224,224,3];input_2:[1,96,96,3];output_1:[1,100];output_2:[1,1000]'
   -  **FoldRelu**: (Boolean) If True, the Relu will be fold to Conv when use ExtendedQuantFormat. The default is False.
   -  **CalibDataSize**: (Int) This parameter controls how many data are used for calibration. The default to using all the data in the calibration dataloader.
   -  **CalibOptimizeMem**: (Boolean) If True, caches intermediate data of activations on disk to reduce the memory consumption during calibration. This option is only effective for the PowerOfTwoMethod.MinMSE method. The default is True.
   -  **CalibWorkerNum**: (Int) This parameter controls how many workers (processes) to collect data. The more workers there are, the less time it takes, but the more memory it consumes (because each worker requires independent memory space). It supports all methods except for CalibrationMethod.MinMax and PowerOfTwoMethod.NonOverflow. The default is 1.
   -  **SaveTensorHistFig**: (Boolean) If True, save the tensor histogram to the file 'tensor_hist' in the working directory. The default is False.
   -  **QuantizeAllOpTypes**: (Boolean) If True, all operation types will be quantized. In the BF16 config, the default is True, while for others, the default is False.
   -  **WeightsOnly**: (Boolean) If True, only quantize weights of the model. The default is False.
   -  **AlignEltwiseQuantType**: (Boolean) If True, quantize weights of the node with the activation quant type if node type in [Mul, Add, Sub, Div, Min, Max] when quant_format is ExtendedQuantFormat.QDQ and enable_npu_cnn is False and enable_npu_transformer is False. The default is False.
   -  **EnableVaimlBF16**: (Boolean) If True, the bfloat16 quantized model with vitis qdq will be converted to a bfloat16 quantized model with bfloat16 weights stored as float32. Vaiml is the name of a compiler, the bfloat16 quantized model can be directly deployed on the compiler if the parameter is True. The default is False.
   -  **UseMatMulNBits**: (Boolean) If True, only quantize weights with nbits for MatMul of the model. The default is False.
   -  **MatMulNBitsParams**: (Dictionary) A parameter used to specify the settings for MatMulNBits Quantizer:

      -  **Algorithm**: (str) The algorithm in MatMulNBits Quantization determines which algorithm ("DEFAULT", "GPTQ", "HQQ") to be used to quantize weights. The default is "DEFAULT".
      -  **GroupSize**: (int) The block size in MatMulNBits Quantization determines how many weights share a scale. The default is 128.
      -  **Symmetric**: (Boolean) If True, symmetrize quantization for weights. The default is True.
      -  **Bits**: (int) The target bits to quantize. Only 4b quantization is supported for inference, additional bits support is planned.
      -  **AccuracyLevel**: (int) The quantization level of input, can be: 0(unset), 1(fp32), 2(fp16), 3(bf16), or 4(int8). The default is 0.

   -  **EvalMetrics**: (Boolean) If True, enables evaluation of the quantized model by measuring cosine similarity and L2 loss. The default is False.
   -  **EvalDataReader**: (DataReader) This parameter is used only when EvalMetrics is set to True. It allows the user to provide a custom data reader for evaluating the quantized model's cosine similarity and L2 loss metrics against the float model.
   -  **TmpDir**: (String) Specifies the directory used to cache intermediate files. The default value is None, in which case the system temporary directory will be used for the caching. This argument can be set for either absolute or relative path.
   -  **EncryptionAlgorithm**: (String) A parameter used to specify the encryption algorithm for crypto mode, only "AES-256" algorithm is supported currently. The default value is None, which means it will not save any intermediate models/files to disk in crypto mode.


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
