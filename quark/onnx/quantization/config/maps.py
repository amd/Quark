from typing import Any, Dict, List, Optional, Tuple, Union

import onnx
from onnxruntime.quantization.calibrate import CalibrationMethod
from onnxruntime.quantization.quant_utils import QuantFormat

from quark.onnx.calibration.methods import LayerWiseMethod, PowerOfTwoMethod
from quark.onnx.quant_utils import ExtendedQuantFormat, recursive_update
from quark.shares.utils.log import ScreenLogger

from .config import QConfig
from .data_type import DataType, Int8, UInt8
from .spec import CalibMethod, QuantGranularity, ScaleType, XInt8Spec

logger = ScreenLogger(__name__)


# TODO: The _map_specific_layer_config function is meant to map the new API specific_layer_config to the old one MixedPrecisionTensor and to maintain compatibility between them. In the future, both this mapping function and the old API will be removed.
def _map_specific_layer_config(
    specific_layer_config: dict[DataType, list[str]], model_input: str
) -> dict[DataType, list[str]]:
    """
    Map new API `specific_layer_config` to the old `MixedPrecisionTensor` format.

    This function loads the ONNX model and traverses its graph nodes. For each layer
    specified in the `specific_layer_config`, it collects the corresponding input
    and output tensor names, and maps them to the expected old API format.

    Args:
        specific_layer_config (dict): Mapping from data types to lists of layer names.
        model_input (str): Path to the ONNX model file.

    Returns:
        dict: A dictionary mapping old-format data types to lists of tensor names.
    """
    if specific_layer_config is None or len(specific_layer_config) == 0:
        return {}
    model = onnx.load(model_input)
    tensor_data_type_dict: dict[DataType, list[str]] = dict()
    for data_type in specific_layer_config:
        layer_list = specific_layer_config[data_type]
        for layer in layer_list:
            for node in model.graph.node:
                if node.name == layer:
                    tmp_list = list(node.input) + list(node.output)
                    if data_type.map_onnx_format not in tensor_data_type_dict:
                        tensor_data_type_dict[data_type.map_onnx_format] = []
                    tensor_data_type_dict[data_type.map_onnx_format] += tmp_list

    return tensor_data_type_dict


# TODO: The _map_layer_type_config function is meant to map the new API layer_type_config to the old one MixedPrecisionTensor and to maintain compatibility between them. In the future, both this mapping function and the old API will be removed.
def _map_layer_type_config(
    layer_type_config: dict[DataType | None, list[str]], model_input: str
) -> tuple[dict[DataType, list[str]], list[str]]:
    """
    Map new API `layer_type_config` to the old `MixedPrecisionTensor` format.

    This function loads the ONNX model and traverses its graph nodes.
    It collects input and output tensors for all nodes whose op_type matches
    the ones in `layer_type_config`, and maps them to the old API format.

    Args:
        layer_type_config (dict): Mapping from data types to lists of operator types.
                                  If `None` is provided for the data type, the nodes
                                  are marked for exclusion instead.
        model_input (str): Path to the ONNX model file.

    Returns:
        tuple:
            - dict: Mapping from old-format data types to lists of tensor names.
            - list: A list of node names to exclude.
    """
    if layer_type_config is None or len(layer_type_config) == 0:
        return {}, []
    model = onnx.load(model_input)
    nodes_to_exclude = []
    tensor_data_type_dict: dict[DataType, list[str]] = dict()
    for data_type in layer_type_config:
        op_types = layer_type_config[data_type]
        if data_type is None:
            for node in model.graph.node:
                if node.op_type in op_types:
                    nodes_to_exclude.append(node.name)
        else:
            op_type_tensor_list = []
            for node in model.graph.node:
                if node.op_type in op_types:
                    op_type_tensor_list += list(node.input)
                    op_type_tensor_list += list(node.output)
            if data_type not in tensor_data_type_dict:
                tensor_data_type_dict[data_type.map_onnx_format] = []
            tensor_data_type_dict[data_type.map_onnx_format] += op_type_tensor_list

    return tensor_data_type_dict, nodes_to_exclude


# TODO: The _map_calibration_method function is meant to map the new calibration method to the old one and to maintain compatibility between them. In the future, both this mapping function and the old calibration method will be removed.
def _map_calibration_method(
    calibrate_method: CalibMethod, scale_type: ScaleType
) -> Union[CalibrationMethod, LayerWiseMethod, PowerOfTwoMethod]:
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
    activation_tensor_instance = q_config.global_config.activation
    weight_tensor_instance = q_config.global_config.weight
    if activation_tensor_instance.scale_type == ScaleType.Float32:
        if activation_tensor_instance.calibration_method == CalibMethod.MinMSE:
            logger.warning(
                "You must use one of (CalibMethod.MinMax, CalibMethod.Percentile, CalibMethod.Entropy, CalibMethod.LayerwisePercentile) when using ScaleType.Float32; otherwise, deployment cannot proceed."
            )
    if weight_tensor_instance.scale_type == ScaleType.Float32:
        if weight_tensor_instance.calibration_method == CalibMethod.MinMSE:
            logger.warning(
                "You must use one of (CalibMethod.MinMax, CalibMethod.Percentile, CalibMethod.Entropy, CalibMethod.LayerwisePercentile) when using ScaleType.Float32; otherwise, deployment cannot proceed."
            )

    if activation_tensor_instance.scale_type == ScaleType.PowerOf2:
        if activation_tensor_instance.calibration_method not in [CalibMethod.MinMSE, CalibMethod.MinMax]:
            logger.warning(
                "You must use one of (CalibMethod.MinMSE, CalibMethod.MinMax) when using ScaleType.PowerOf2; otherwise, deployment cannot proceed."
            )
    if weight_tensor_instance.scale_type == ScaleType.PowerOf2:
        if weight_tensor_instance.calibration_method not in [CalibMethod.MinMSE, CalibMethod.MinMax]:
            logger.warning(
                "You must use one of (CalibMethod.MinMSE, CalibMethod.MinMax) when using ScaleType.PowerOf2; otherwise, deployment cannot proceed."
            )
    mapping["calibrate_method"] = _map_calibration_method(
        activation_tensor_instance.calibration_method, activation_tensor_instance.scale_type
    )
    mapping["activation_type"] = activation_tensor_instance.data_type
    mapping["weight_type"] = weight_tensor_instance.data_type
    if activation_tensor_instance.data_type in [Int8, UInt8] and weight_tensor_instance.data_type in [  # type: ignore
        Int8,
        UInt8,
    ]:
        mapping["quant_format"] = QuantFormat.QDQ
    else:
        mapping["quant_format"] = ExtendedQuantFormat.QDQ
    if weight_tensor_instance.quant_granularity == QuantGranularity.Channel:
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
    mapping["extra_options"]["ActivationSymmetric"] = activation_tensor_instance.symmetric
    mapping["extra_options"]["WeightSymmetric"] = weight_tensor_instance.symmetric
    if "MixedPrecisionTensor" in q_config.extra_options:
        mapping["extra_options"]["MixedPrecisionTensor"] = q_config.extra_options["MixedPrecisionTensor"]
    else:
        mapping["extra_options"]["MixedPrecisionTensor"] = dict()
    recursive_update(
        mapping["extra_options"]["MixedPrecisionTensor"],
        _map_specific_layer_config(q_config.specific_layer_config, model_input),
    )
    recursive_update(
        mapping["extra_options"]["MixedPrecisionTensor"],
        _map_layer_type_config(q_config.layer_type_config, model_input)[0],
    )
    if len(mapping["extra_options"]["MixedPrecisionTensor"]) > 0:
        mapping["extra_options"]["SpecificTensorPrecision"] = True
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
        if type(activation_tensor_instance) == XInt8Spec and type(weight_tensor_instance) == XInt8Spec:
            mapping["extra_options"]["EnableNPUCnn"] = True
        else:
            mapping["extra_options"]["EnableNPUCnn"] = False
    return mapping
