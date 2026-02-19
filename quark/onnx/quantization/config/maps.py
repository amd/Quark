#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Any

import onnx
from onnx import NodeProto
from onnxruntime.quantization.calibrate import CalibrationMethod
from onnxruntime.quantization.quant_utils import QuantFormat

from quark.onnx.calibration.methods import ExtendedCalibrationMethod, LayerWiseMethod, PowerOfTwoMethod
from quark.onnx.quantization.quant_utils import (
    ExtendedQuantFormat,
    get_all_target_nodes,
    recursive_update,
)
from quark.shares.utils.log import ScreenLogger

from .config import QConfig
from .data_type import DataType, Int8, UInt8
from .spec import (
    CalibMethod,
    MX4Spec,
    MX6Spec,
    MX9Spec,
    MXFP4E2M1Spec,
    MXFP6E2M3Spec,
    MXFP6E3M2Spec,
    MXFP8E4M3Spec,
    MXFP8E5M2Spec,
    MXInt8Spec,
    QLayerConfig,
    QTensorConfig,
    QuantGranularity,
    ScaleType,
    XInt8Spec,
)

logger = ScreenLogger(__name__)

QCONFIG_ALL_PARAMS = {
    "global_config",
    "specific_layer_config",
    "layer_type_config",
    "exclude",
    "algo_config",
    "use_external_data_format",
    "extra_options",
    "PreprocessYAML",
    "OpTypesToQuantize",
    "ExtraOpTypesToQuantize",
    "ExecutionProviders",
    "OptimizeModel",
    "ConvertFP16ToFP32",
    "ConvertNCHWToNHWC",
    "DebugMode",
    "CryptoMode",
    "PrintSummary",
    "IgnoreWarnings",
    "LogSeverityLevel",
    "ActivationScaled",
    "WeightScaled",
    "QuantizeFP16",
    "UseFP32Scale",
    "UseUnsignedReLU",
    "QuantizeBias",
    "Int32Bias",
    "Int16Bias",
    "RemoveInputInit",
    "SimplifyModel",
    "EnableSubgraph",
    "ForceQuantizeNoInputCheck",
    "MatMulConstBOnly",
    "AddQDQPairToWeight",
    "OpTypesToExcludeOutputQuantization",
    "DedicatedQDQPair",
    "QDQOpTypePerChannelSupportToAxis",
    "CalibTensorRangeSymmetric",
    "CalibMovingAverage",
    "CalibMovingAverageConstant",
    "Percentile",
    "LWPMetric",
    "ActivationBitWidth",
    "PercentileCandidates",
    "UseRandomData",
    "RandomDataReaderInputShape",
    "RandomDataReaderInputDataRange",
    "Int16Scale",
    "MinMSEModePof2Scale",
    "ConvertOpsetVersion",
    "ConvertBNToConv",
    "ConvertReduceMeanToGlobalAvgPool",
    "SplitLargeKernelPool",
    "ConvertSplitToSlice",
    "FuseInstanceNorm",
    "FuseL2Norm",
    "FuseGelu",
    "FuseLayerNorm",
    "ConvertClipToRelu",
    "SimulateDPU",
    "ConvertLeakyReluToDPUVersion",
    "ConvertSigmoidToHardSigmoid",
    "ConvertHardSigmoidToDPUVersion",
    "ConvertAvgPoolToDPUVersion",
    "ConvertClipToDPUVersion",
    "ConvertReduceMeanToDPUVersion",
    "ConvertSoftmaxToDPUVersion",
    "NPULimitationCheck",
    "MaxLoopNum",
    "AdjustShiftCut",
    "AdjustShiftBias",
    "AdjustShiftRead",
    "AdjustShiftWrite",
    "AdjustHardSigmoid",
    "AdjustShiftSwish",
    "AlignConcat",
    "AlignPool",
    "AlignPad",
    "AlignSlice",
    "AlignTranspose",
    "AlignReshape",
    "AdjustBiasScale",
    "TensorsRangeFile",
    "ReplaceClip6Relu",
    "CopySharedInit",
    "CopyBiasInit",
    "RemoveQDQConvClip",
    "RemoveQDQConvRelu",
    "RemoveQDQConvLeakyRelu",
    "RemoveQDQConvPRelu",
    "RemoveQDQConvGelu",
    "RemoveQDQMulAdd",
    "RemoveQDQBetweenOps",
    "RemoveQDQInstanceNorm",
    "FoldBatchNorm",
    "BF16WithClip",
    "BF16QDQToCast",
    "FixShapes",
    "FoldRelu",
    "CalibDataSize",
    "CalibOptimizeMem",
    "CalibWorkerNum",
    "SaveTensorHistFig",
    "QuantizeAllOpTypes",
    "WeightsOnly",
    "AlignEltwiseQuantType",
    "EnableVaimlBF16",
    "UseMatMulNBits",
    "MatMulNBitsParams",
    "EvalMetrics",
    "EvalDataReader",
    "TmpDir",
    "EncryptionAlgorithm",
    "WeightCalibrateMethod",
    "MinMSEModeFloatScale",
}


def _check_q_config(q_config: QConfig) -> None:
    """
    Check whether the parameters in QConfig are invalid.
    """

    if len(q_config.extra_options) > 0:
        if "extra_options" not in q_config.extra_options:
            logger.warning(
                "Detected parameters that should be placed under extra_options. Please move them to extra_options; direct usage will be deprecated in the next release."
            )
        else:
            assert q_config.extra_options["extra_options"] is None or isinstance(
                q_config.extra_options["extra_options"], dict
            ), "'extra_options' must be a dict."
            q_config.extra_options.update(q_config.extra_options.pop("extra_options", {}) or {})

    for k, _ in q_config.__dict__.items():
        if k not in QCONFIG_ALL_PARAMS:
            logger.warning(f"{k} is an invalid parameter in QCONFIG. Please check it.")
    for k in q_config.extra_options:
        if k not in QCONFIG_ALL_PARAMS:
            logger.warning(f"{k} is an invalid parameter in QCONFIG. Please check it.")


DEFAULT_MICROEXPONENTS_PARAMS = {
    "bfp_method": "to_bfp_prime",
    "axis": 1,
    "bit_width": 13,
    "block_size": 16,
    "sub_block_size": 2,
    "sub_block_shift_bits": 1,
    "rounding_mode": 2,
}

DEFAULT_MICROSCALING_PARAMS = {
    "element_dtype": "int8",
    "axis": 1,
    "block_size": 32,
    "rounding_mode": 2,
}


def _check_global_config(qlayer_config: QLayerConfig) -> QLayerConfig:
    """
    Validate and normalize the global layer quantization configuration.

    This function checks the consistency of activation-related settings in
    `QLayerConfig`, ensuring that only `input_tensors` is used for specifying
    activation quantization. If the deprecated `activation` field is provided,
    it will be mapped to `input_tensors` with a warning. The function also
    enforces that a valid weight configuration is present.

    :param QLayerConfig qlayer_config: Global layer quantization configuration.
    :return: The validated and possibly normalized configuration.
    :raises ValueError: If both `activation` and `input_tensors` are provided,
                        if neither is provided, or if `weight` is not specified.

    """
    input_tensors = qlayer_config.input_tensors
    activation = qlayer_config.activation
    weight = qlayer_config.weight

    if activation is not None and input_tensors is not None:
        raise ValueError("Both `activation` and `input_tensors` are provided. Please just use `input_tensors`.")

    if activation is None and input_tensors is None:
        raise ValueError("You must specify `input_tensors`.")

    if activation is not None and input_tensors is None:
        logger.warning(
            "The api `activation` has been replaced with `input_tensors` and will be removed in the next release."
        )
        qlayer_config.input_tensors = qlayer_config.activation

    if weight is None:
        raise ValueError("You must specify either `weight`.")

    return qlayer_config


def _check_qlayer_config(qlayer_config: QLayerConfig) -> None:
    """
    Validate the per-layer quantization configuration.

    This function ensures that deprecated fields are not used in an
    incompatible way. Specifically, it checks that the legacy `activation`
    field is not provided without the newer `input_tensors` field, enforcing
    the current API requirements.

    :param QLayerConfig qlayer_config: Layer-level quantization configuration.
    :raises ValueError: If the deprecated `activation` field is used without
                        specifying `input_tensors`.

    """
    input_tensors = qlayer_config.input_tensors
    activation = qlayer_config.activation

    if activation is not None and input_tensors is None:
        raise ValueError(
            "The api `activation` has been replaced with `input_tensors` and will be removed in the next release."
        )


def _map_mx_config(
    input_tensors_instance: QTensorConfig, weight_instance: QTensorConfig, extra_options: dict[str, Any]
) -> dict[str, Any]:
    if type(input_tensors_instance) == MX4Spec and type(weight_instance) == MX4Spec:
        if "BFPAttributes" not in extra_options:
            extra_options["BFPAttributes"] = {**DEFAULT_MICROEXPONENTS_PARAMS, "bit_width": 11}
    if type(input_tensors_instance) == MX6Spec and type(weight_instance) == MX6Spec:
        if "BFPAttributes" not in extra_options:
            extra_options["BFPAttributes"] = {**DEFAULT_MICROEXPONENTS_PARAMS, "bit_width": 13}
    if type(input_tensors_instance) == MX9Spec and type(weight_instance) == MX9Spec:
        if "BFPAttributes" not in extra_options:
            extra_options["BFPAttributes"] = {**DEFAULT_MICROEXPONENTS_PARAMS, "bit_width": 16}
    if type(input_tensors_instance) == MXFP4E2M1Spec and type(weight_instance) == MXFP4E2M1Spec:
        if "MXAttributes" not in extra_options:
            extra_options["MXAttributes"] = {**DEFAULT_MICROSCALING_PARAMS, "element_dtype": "fp4_e2m1"}
    if type(input_tensors_instance) == MXFP6E3M2Spec and type(weight_instance) == MXFP6E3M2Spec:
        if "MXAttributes" not in extra_options:
            extra_options["MXAttributes"] = {**DEFAULT_MICROSCALING_PARAMS, "element_dtype": "fp6_e3m2"}
    if type(input_tensors_instance) == MXFP6E2M3Spec and type(weight_instance) == MXFP6E2M3Spec:
        if "MXAttributes" not in extra_options:
            extra_options["MXAttributes"] = {**DEFAULT_MICROSCALING_PARAMS, "element_dtype": "fp6_e2m3"}
    if type(input_tensors_instance) == MXFP8E5M2Spec and type(weight_instance) == MXFP8E5M2Spec:
        if "MXAttributes" not in extra_options:
            extra_options["MXAttributes"] = {**DEFAULT_MICROSCALING_PARAMS, "element_dtype": "fp8_e5m2"}
    if type(input_tensors_instance) == MXFP8E4M3Spec and type(weight_instance) == MXFP8E4M3Spec:
        if "MXAttributes" not in extra_options:
            extra_options["MXAttributes"] = {**DEFAULT_MICROSCALING_PARAMS, "element_dtype": "fp8_e4m3"}
    if type(input_tensors_instance) == MXInt8Spec and type(weight_instance) == MXInt8Spec:
        if "MXAttributes" not in extra_options:
            extra_options["MXAttributes"] = {**DEFAULT_MICROSCALING_PARAMS, "element_dtype": "int8"}
    return extra_options


def _update_tensor_quant_config_dict(
    all_init_names: set[str],
    node: NodeProto,
    qlayer_config: QLayerConfig,
    tensor_quant_config_dict: dict[str, list[dict[str, DataType | bool]]],
) -> dict[str, list[dict[str, DataType | bool]]]:
    """
    Update tensor-level quantization configuration for a given ONNX node.

    This function populates or updates entries in `tensor_quant_config_dict`
    based on the quantization settings specified in `QLayerConfig`. It assigns
    quantization data type and symmetry information to the node’s input, weight,
    bias, and output tensors when the corresponding configurations are provided.

    :param set[str] all_init_names: All initializer names of the ONNX model.
    :param NodeProto node: The ONNX node whose tensors are being processed.
    :param QLayerConfig qlayer_config: Layer-level quantization configuration.
    :param dict tensor_quant_config_dict: Mapping from tensor names to tensor
        quantization configuration entries.
    :return: The updated tensor quantization configuration dictionary, mapping tensor
        names to their quantization configurations.

    """
    i = 0
    while i < len(node.input) and (node.input[i] not in all_init_names):
        if qlayer_config.input_tensors is not None:
            tensor_quant_config_dict[node.input[i]] = [
                {
                    "quant_type": qlayer_config.input_tensors.data_type.map_onnx_format,  # type: ignore
                    "symmetric": qlayer_config.input_tensors.symmetric,
                }
            ]
        i += 1

    if len(node.input) >= i + 1 and qlayer_config.weight is not None:
        tensor_quant_config_dict[node.input[i]] = [
            {"quant_type": qlayer_config.weight.data_type.map_onnx_format, "symmetric": qlayer_config.weight.symmetric}  # type: ignore
        ]

    if len(node.input) >= i + 2 and qlayer_config.bias is not None:
        tensor_quant_config_dict[node.input[i + 1]] = [
            {"quant_type": qlayer_config.bias.data_type.map_onnx_format, "symmetric": qlayer_config.bias.symmetric}  # type: ignore
        ]

    if len(node.output) >= 1 and qlayer_config.output_tensors is not None:
        for i in range(len(node.output)):
            tensor_quant_config_dict[node.output[i]] = [
                {
                    "quant_type": qlayer_config.output_tensors.data_type.map_onnx_format,  # type: ignore
                    "symmetric": qlayer_config.output_tensors.symmetric,
                }
            ]

    return tensor_quant_config_dict


# TODO: The _map_specific_layer_config function is meant to map the new API specific_layer_config to the old one MixedPrecisionTensor and to maintain compatibility between them. In the future, both this mapping function and the old API will be removed.
def _map_specific_layer_config(
    specific_layer_config: dict[QLayerConfig, list[str]], model_input: str
) -> dict[str, list[dict[str, DataType | bool]]]:
    """
    Map layer-specific quantization configurations to tensor-level entries.

    This function processes a `specific_layer_config` dictionary that maps
    `QLayerConfig` instances to lists of layer names in an ONNX model. For each
    specified layer, it identifies the corresponding ONNX nodes and updates the
    tensor-level quantization configuration for inputs, weights, biases, and outputs.

    :param dict[QLayerConfig, list[str]] specific_layer_config: Mapping from layer-level
        quantization configurations to lists of target layer names.
    :param str model_input: Path to the ONNX model file.
    :return: A tensor-level quantization configuration dictionary, where keys are tensor
        names and values are lists of dictionaries specifying quantization type and
        symmetry.
    """
    if specific_layer_config is None or len(specific_layer_config) == 0:
        return {}
    model = onnx.load(model_input)
    all_init_names = {init.name for init in model.graph.initializer}
    tensor_quant_config_dict: dict[str, list[dict[str, DataType | bool]]] = dict()
    for qlayer_config, layer_list in specific_layer_config.items():
        _check_qlayer_config(qlayer_config)

        target_nodes = get_all_target_nodes(model, layer_list)
        for node in model.graph.node:
            if node.name in target_nodes:
                tensor_quant_config_dict = _update_tensor_quant_config_dict(
                    all_init_names, node, qlayer_config, tensor_quant_config_dict
                )

    return tensor_quant_config_dict


# TODO: The _map_layer_type_config function is meant to map the new API layer_type_config to the old one MixedPrecisionTensor and to maintain compatibility between them. In the future, both this mapping function and the old API will be removed.
def _map_layer_type_config(
    layer_type_config: dict[DataType | None, list[str]], model_input: str
) -> tuple[dict[str, list[dict[str, DataType | bool]]], list[str]]:
    """
    Map operator-type-based quantization configurations to tensor-level entries
    and collect nodes to exclude.

    This function processes a `layer_type_config` dictionary that maps either
    a `QLayerConfig` (for quantization) or `None` (for exclusion) to lists of
    operator types. For each node in the ONNX model whose `op_type` matches the
    specified types, it either updates the tensor-level quantization configuration
    or adds the node name to the exclusion list.

    :param dict[DataType | None, list[str]] layer_type_config: Mapping from `QLayerConfig`
        or `None` to lists of ONNX operator types. If the key is `None`, the nodes are
        excluded from quantization.
    :param str model_input: Path to the ONNX model file.
    :return: A tuple containing:
        - A tensor-level quantization configuration dictionary, where keys are tensor
          names and values are lists of dictionaries specifying quantization type and
          symmetry.
        - A list of node names to exclude from quantization.
    """
    if layer_type_config is None or len(layer_type_config) == 0:
        return {}, []
    model = onnx.load(model_input)
    all_init_names = {init.name for init in model.graph.initializer}
    nodes_to_exclude = []
    tensor_quant_config_dict: dict[str, list[dict[str, DataType | bool]]] = dict()
    for qlayer_config, op_types in layer_type_config.items():
        if qlayer_config is None:
            for node in model.graph.node:
                if node.op_type in op_types:
                    nodes_to_exclude.append(node.name)
        else:
            _check_qlayer_config(qlayer_config)
            for node in model.graph.node:
                if node.op_type in op_types:
                    tensor_quant_config_dict = _update_tensor_quant_config_dict(
                        all_init_names, node, qlayer_config, tensor_quant_config_dict
                    )
    return tensor_quant_config_dict, nodes_to_exclude


def _map_mixed_precision_tensors(extra_options: dict[str, Any]) -> None:
    """
    Map the legacy format 'MixedPrecisionTensor' to 'TensorQuantOverrides'.

    :param Dict[str, Any] extra_options: Extra options for quantization.

    """
    if extra_options.get("MixedPrecisionTensor") is None:
        return None

    logger.warning(
        "The option 'MixedPrecisionTensor' will be deprecated in future versions, "
        "please use 'TensorQuantOverrides' instead."
    )

    if extra_options.get("TensorQuantOverrides") is None:
        extra_options["TensorQuantOverrides"] = {}

    for k, v in extra_options["MixedPrecisionTensor"].items():
        for tensor_name in v:
            if tensor_name in extra_options["TensorQuantOverrides"]:
                for override in extra_options["TensorQuantOverrides"][tensor_name]:
                    assert isinstance(override, dict), f"The {override} should be a dict."
                    override["quant_type"] = k
            else:
                extra_options["TensorQuantOverrides"][tensor_name] = [{"quant_type": k}]

    extra_options.pop("MixedPrecisionTensor")


# TODO: The _map_activation_calibration_method function is meant to map the new calibration method to the old one and to maintain compatibility between them. In the future, both this mapping function and the old calibration method will be removed.
def _map_activation_calibration_method(
    calibrate_method: CalibMethod, scale_type: ScaleType
) -> CalibrationMethod | LayerWiseMethod | PowerOfTwoMethod:
    """
    Map a new API calibration method and scale type to the old API format.

    Args:
        calibrate_method (CalibMethod): The calibration method from the new API.
        scale_type (ScaleType): The scale type (e.g., Float32, PowerOf2).

    Returns:
        Any: The corresponding calibration method in the old API.

    Raises:
        ValueError: If the calibration method or scale type is invalid.
    """
    if calibrate_method == CalibMethod.MinMax and scale_type == ScaleType.Float32:
        return CalibrationMethod.MinMax
    elif calibrate_method == CalibMethod.Percentile:
        return CalibrationMethod.Percentile
    elif calibrate_method == CalibMethod.LayerwisePercentile:
        return LayerWiseMethod.LayerWisePercentile
    elif calibrate_method == CalibMethod.Distribution:
        return CalibrationMethod.Distribution
    elif calibrate_method == CalibMethod.MinMSE:
        return PowerOfTwoMethod.MinMSE
    elif calibrate_method == CalibMethod.MinMax and scale_type == ScaleType.PowerOf2:
        return PowerOfTwoMethod.NonOverflow
    elif calibrate_method == CalibMethod.Entropy:
        return CalibrationMethod.Entropy
    else:
        raise ValueError("The calibration method id invalid.")


# TODO: The _map_weight_calibration_method function is meant to map the new calibration method to the old one and to maintain compatibility between them. In the future, both this mapping function and the old calibration method will be removed.
def _map_weight_calibration_method(calibrate_method: CalibMethod, extra_options: dict[str, Any]) -> dict[str, Any]:
    """
    Map a new API calibration method and scale type to the old API format.

    Args:
        calibrate_method (CalibMethod): The calibration method from the new API.
        extra_options (Dict[str, Any]): The kwargs of QConfig.

    Returns:
        extra_options (Dict[str, Any]): The kwargs of QConfig.
    """
    if calibrate_method == CalibMethod.MinMSE:
        extra_options["WeightCalibrateMethod"] = ExtendedCalibrationMethod.MinMSE
    else:
        extra_options["WeightCalibrateMethod"] = CalibrationMethod.MinMax
    return extra_options


# TODO: The _map_q_config function is meant to map the new API to the old one and to maintain compatibility between them. In the future, both this mapping function and the old API will be removed.
def _map_q_config(q_config: QConfig, model_input: str) -> dict[str, Any]:
    """
    Map a full quantization config from the new API to the old API format.

    This function handles global quantization specs for activations and weights,
    calibration methods, scale types, granularity, exclusion rules, and additional
    options. It also validates configuration combinations and logs warnings if
    invalid or unsupported settings are detected.

    Args:
        q_config: A configuration object from the new API.
        model_input (str): Path to the ONNX model file.

    Returns:
        Dict[str, Any]: A dictionary in the old API format, containing all necessary
                        quantization settings and extra options.
    """
    mapping = dict()
    q_config.global_config = _check_global_config(q_config.global_config)
    input_tensors_instance = q_config.global_config.input_tensors
    weight_instance = q_config.global_config.weight
    assert input_tensors_instance is not None and weight_instance is not None
    if input_tensors_instance.scale_type == ScaleType.Float32:
        if input_tensors_instance.calibration_method == CalibMethod.MinMSE:
            logger.warning(
                "You must use one of (CalibMethod.MinMax, CalibMethod.Percentile, CalibMethod.Entropy, CalibMethod.LayerwisePercentile) when using ScaleType.Float32; otherwise, deployment cannot proceed."
            )
    if weight_instance.scale_type == ScaleType.Float32:
        if weight_instance.calibration_method == CalibMethod.MinMSE:
            logger.warning(
                "You must use one of (CalibMethod.MinMax, CalibMethod.Percentile, CalibMethod.Entropy, CalibMethod.LayerwisePercentile) when using ScaleType.Float32; otherwise, deployment cannot proceed."
            )

    if input_tensors_instance.scale_type == ScaleType.PowerOf2:
        if input_tensors_instance.calibration_method not in [CalibMethod.MinMSE, CalibMethod.MinMax]:
            logger.warning(
                "You must use one of (CalibMethod.MinMSE, CalibMethod.MinMax) when using ScaleType.PowerOf2; otherwise, deployment cannot proceed."
            )
    if weight_instance.scale_type == ScaleType.PowerOf2:
        if weight_instance.calibration_method not in [CalibMethod.MinMSE, CalibMethod.MinMax]:
            logger.warning(
                "You must use one of (CalibMethod.MinMSE, CalibMethod.MinMax) when using ScaleType.PowerOf2; otherwise, deployment cannot proceed."
            )
    mapping["calibrate_method"] = _map_activation_calibration_method(
        input_tensors_instance.calibration_method, input_tensors_instance.scale_type
    )
    mapping["activation_type"] = input_tensors_instance.data_type
    mapping["weight_type"] = weight_instance.data_type
    if input_tensors_instance.data_type in [Int8, UInt8] and weight_instance.data_type in [  # type: ignore
        Int8,
        UInt8,
    ]:
        mapping["quant_format"] = QuantFormat.QDQ
    else:
        mapping["quant_format"] = ExtendedQuantFormat.QDQ
    if weight_instance.quant_granularity == QuantGranularity.Channel:
        mapping["per_channel"] = True
    else:
        mapping["per_channel"] = False
    mapping["nodes_to_exclude"] = []
    mapping["subgraphs_to_exclude"] = []
    if q_config.exclude is not None and len(q_config.exclude) > 0:
        for tmp in q_config.exclude:
            if isinstance(tmp, str):
                mapping["nodes_to_exclude"].append(tmp)  # type: ignore
            if isinstance(tmp, tuple):
                mapping["subgraphs_to_exclude"].append(tmp)
    mapping["use_external_data_format"] = q_config.use_external_data_format
    mapping["extra_options"] = q_config.extra_options
    if (
        input_tensors_instance.scale_type == ScaleType.PowerOf2
        and input_tensors_instance.calibration_method != weight_instance.calibration_method
    ):
        logger.warning(
            "If the weight’s scale type is power-of-2, its calibration method will automatically align with the activation’s calibration method."
        )
    if "WeightCalibrateMethod" in q_config.extra_options and (
        weight_instance.scale_type == ScaleType.PowerOf2 or weight_instance.scale_type == ScaleType.Int16
    ):
        del mapping["extra_options"]["WeightCalibrateMethod"]  # type: ignore
        logger.warning("The WeightCalibrateMethod parameter can be used only when the scale type if float32.")
    if "WeightCalibrateMethod" not in q_config.extra_options and weight_instance.scale_type == ScaleType.Float32:
        mapping["extra_options"] = _map_weight_calibration_method(
            weight_instance.calibration_method, mapping["extra_options"]
        )
    mapping["extra_options"]["ActivationSymmetric"] = input_tensors_instance.symmetric
    mapping["extra_options"]["WeightSymmetric"] = weight_instance.symmetric
    if "TensorQuantOverrides" in q_config.extra_options:
        mapping["extra_options"]["TensorQuantOverrides"] = q_config.extra_options["TensorQuantOverrides"]
    else:
        mapping["extra_options"]["TensorQuantOverrides"] = dict()
    recursive_update(
        mapping["extra_options"]["TensorQuantOverrides"],
        _map_specific_layer_config(q_config.specific_layer_config, model_input),
    )
    recursive_update(
        mapping["extra_options"]["TensorQuantOverrides"],
        _map_layer_type_config(q_config.layer_type_config, model_input)[0],
    )
    if "MixedPrecisionTensor" in q_config.extra_options:
        mapping["extra_options"]["MixedPrecisionTensor"] = q_config.extra_options["MixedPrecisionTensor"]
    else:
        mapping["extra_options"]["MixedPrecisionTensor"] = dict()
    if len(mapping["extra_options"]["MixedPrecisionTensor"]) > 0:
        mapping["extra_options"]["SpecificTensorPrecision"] = True
        _map_mixed_precision_tensors(mapping["extra_options"])
    else:
        mapping["extra_options"]["SpecificTensorPrecision"] = False
    mapping["nodes_to_exclude"] += _map_layer_type_config(q_config.layer_type_config, model_input)[1]  # type: ignore
    if "InputNodes" in q_config.extra_options:
        mapping["extra_options"]["InputNodes"] = q_config.extra_options["InputNodes"]
    else:
        mapping["extra_options"]["InputNodes"] = []
    if "OutputNodes" in q_config.extra_options:
        mapping["extra_options"]["OutputNodes"] = q_config.extra_options["OutputNodes"]
    else:
        mapping["extra_options"]["OutputNodes"] = []
    if "OpTypesToQuantize" in q_config.extra_options:
        mapping["extra_options"]["OpTypesToQuantize"] = q_config.extra_options["OpTypesToQuantize"]
    else:
        mapping["extra_options"]["OpTypesToQuantize"] = []
    if "NodesToQuantize" in q_config.extra_options:
        mapping["extra_options"]["NodesToQuantize"] = q_config.extra_options["NodesToQuantize"]
    else:
        mapping["extra_options"]["NodesToQuantize"] = []
    if "ExtraOpTypesToQuantize" in q_config.extra_options:
        mapping["extra_options"]["ExtraOpTypesToQuantize"] = q_config.extra_options["ExtraOpTypesToQuantize"]
    else:
        mapping["extra_options"]["ExtraOpTypesToQuantize"] = []
    if "ExecutionProviders" in q_config.extra_options:
        mapping["extra_options"]["ExecutionProviders"] = q_config.extra_options["ExecutionProviders"]
    else:
        mapping["extra_options"]["ExecutionProviders"] = ["CPUExecutionProvider"]
    if "OptimizeModel" in q_config.extra_options:
        mapping["extra_options"]["OptimizeModel"] = q_config.extra_options["OptimizeModel"]
    else:
        mapping["extra_options"]["OptimizeModel"] = True
    if "ConvertFP16ToFP32" in q_config.extra_options:
        mapping["extra_options"]["ConvertFP16ToFP32"] = q_config.extra_options["ConvertFP16ToFP32"]
    else:
        mapping["extra_options"]["ConvertFP16ToFP32"] = False
    if "ConvertNCHWToNHWC" in q_config.extra_options:
        mapping["extra_options"]["ConvertNCHWToNHWC"] = q_config.extra_options["ConvertNCHWToNHWC"]
    else:
        mapping["extra_options"]["ConvertNCHWToNHWC"] = False
    if "DebugMode" in q_config.extra_options:
        mapping["extra_options"]["DebugMode"] = q_config.extra_options["DebugMode"]
    else:
        mapping["extra_options"]["DebugMode"] = False
    if "CryptoMode" in q_config.extra_options:
        mapping["extra_options"]["CryptoMode"] = q_config.extra_options["CryptoMode"]
    else:
        mapping["extra_options"]["CryptoMode"] = False
    if "PrintSummary" in q_config.extra_options:
        mapping["extra_options"]["PrintSummary"] = q_config.extra_options["PrintSummary"]
    else:
        mapping["extra_options"]["PrintSummary"] = True
    if "EnableNPUCnn" in q_config.extra_options:
        mapping["extra_options"]["EnableNPUCnn"] = q_config.extra_options["EnableNPUCnn"]
    else:
        if type(input_tensors_instance) == XInt8Spec and type(weight_instance) == XInt8Spec:
            mapping["extra_options"]["EnableNPUCnn"] = True
        else:
            mapping["extra_options"]["EnableNPUCnn"] = False
    mapping["extra_options"] = _map_mx_config(input_tensors_instance, weight_instance, mapping["extra_options"])
    return mapping
